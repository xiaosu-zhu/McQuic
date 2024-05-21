from typing import List, Union, Optional
import logging
from functools import partial
from itertools import repeat
import collections.abc

import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
import random
from fairscale.nn.checkpoint import checkpoint_wrapper

# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_varlen_func
from apex.normalization import FusedRMSNorm as RMSNorm

import transformers.modeling_outputs
from transformers import CLIPTextModel, CLIPProcessor

from mcquic.modules.compressor import Neon
from mcquic.utils.registry import GeneratorRegistry


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


@GeneratorRegistry.register
class GeneratorV3SelfAttention(nn.Module):
    def __init__(
        self,
        channel: int,
        k: int,
        size: List[int],
        denseNorm: bool,
        loadFrom: str,
        qk_norm: bool,
        norm_eps: float,
        *_,
        **__,
    ):
        super().__init__()
        self.compressor = Neon(channel, k, size, denseNorm)
        state_dict = torch.load(loadFrom, map_location="cpu")
        self.compressor.load_state_dict(
            {
                k[len("module._compressor.") :]: v
                for k, v in state_dict["trainer"]["_model"].items()
                if "_lpips" not in k
            }
        )
        for params in self.compressor.parameters():
            params.requires_grad_(False)
        logging.info("Loaded compressor checkpoint from %s.", loadFrom)

        # logging.debug("Start loading clip...")
        # self.text_encoder = CLIPTextModel.from_pretrained(
        #     "openai/clip-vit-base-patch32", local_files_only=True
        # )
        # logging.debug("Loaded clip text model from %s.", "openai/clip-vit-base-patch32")
        # self.text_tokenizer = CLIPProcessor.from_pretrained(
        #     "openai/clip-vit-base-patch32", local_files_only=True
        # )
        # logging.debug("Loaded clip text model from %s.", "openai/clip-vit-base-patch32")
        # for params in self.text_encoder.parameters():
        #     params.requires_grad_(False)

        # clip_text_channels = self.text_encoder.text_model.config.hidden_size

        # NOTE: text_to_first_level: This transforms text embeddings to the first level token
        # NOTE: next_residual_predictor: we only need first (level - 1) codebook, and corresponding canvas.
        # NOTE: remove first dim of codebook, since it is for product quantization
        self.next_residual_predictor = AnyRes_L(
            size[::-1],
            [[4096, 32] for _ in size[::-1]],
            qk_norm=qk_norm,
            norm_eps=norm_eps,
        )
        logging.debug("Created any-res transformer.")

        from mcquic.data.imagenet_classes import IMAGENET2012_LABELS

        self.class_tokens = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(
                    len(IMAGENET2012_LABELS), self.next_residual_predictor.hidden_size
                ),
                std=math.sqrt(2 / (5 * self.next_residual_predictor.hidden_size)),
            )
        )

        self.compressor.eval()
        # self.text_encoder.eval()
        # Cast to bfloat16
        # self.text_encoder.float16()
        # self.text_to_first_level.bfloat16()
        # self.next_residual_predictor.bfloat16()

    def train(self, mode: bool = True):
        retValue = super().train(mode)
        self.compressor.eval()
        # self.text_encoder.eval()
        return retValue

    def forward(self, image, condition: torch.Tensor):
        if self.training:
            ###################### Preparing inputs #########################
            with torch.no_grad():
                # list of [n, 1, h, w], len of list == levels
                # from low resolution to high resolution
                # NOTE: for reflection padding, the input tensor size (`.numel()`) should not exceed 2^32
                # NOTE: therefore, we manually split image into batch 16
                all_forwards_for_residual = list()
                codes = self.compressor.encode(image)
                formerLevel = None
                for level, code in enumerate(codes[:-1]):
                    # list - 1 of [n, c, 2h, 2w]
                    all_forwards_for_residual.append(
                        self.compressor.residual_forward(code, formerLevel, level)
                    )
                    formerLevel = all_forwards_for_residual[-1]

                # input_ids: [B, max_len] int ids, where `49407` for padding
                # attention_mask: [B, max_len] {0, 1}. where `1` for valid, `0` for padding mask
                # batch_encoding = self.text_tokenizer(
                #     text=condition,
                #     return_attention_mask=True,
                #     padding=True,
                #     truncation=True,
                #     return_tensors="pt",
                # )

                # input_ids = batch_encoding.input_ids.to(image.device)
                # attention_mask = batch_encoding.attention_mask.to(image.device)

                # # last_hidden_state: [B, max_len, D]
                # # pooler_output: [B, D]
                # text_embedding: (
                #     transformers.modeling_outputs.BaseModelOutputWithPooling
                # ) = self.text_encoder(
                #     input_ids, attention_mask=attention_mask, return_dict=True
                # )

            # NOTE: remove product quantization artifacts, since we don't use product quantization
            codes = [c.squeeze(1) for c in codes]

            rawPredictions = self.next_residual_predictor(
                [None, *all_forwards_for_residual], self.class_tokens[condition]
            )

            loss = list()
            curIdx = 0
            predictions = list()
            for gt in codes:
                bs, h, w = gt.shape
                pre = rawPredictions[:, curIdx : curIdx + (h * w)]
                pre = pre.permute(0, 2, 1).reshape(bs, -1, h, w)
                predictions.append(pre)
                loss.append(F.cross_entropy(pre, gt, reduction="none"))
                curIdx += h * w

            # loss = [
            #     F.cross_entropy(pre, gt, reduction="none")
            #     for pre, gt in zip([*predictions], codes)
            # ]

            # list of [n, 1, h, w], len of list == levels
            restoredCodes = [
                pre.detach().clone().argmax(1, keepdim=True) for pre in predictions
            ]
            # [n, 1, h, w]
            # restoredCodes.insert(
            #     0, first_level.detach().clone().argmax(1, keepdim=True)
            # )
            with torch.no_grad():
                restored = self.compressor.decode(restoredCodes)

            # first_level: [n, k, h, w]
            # predictions: list of [n, k, h, w], len of list == levels - 1 (give previous embedding, predict next code)
            return (
                [*predictions],
                sum([l.sum() / len(image) for l in loss]),
                codes,
                restored,
                [l.mean() for l in loss],
            )
        else:
            # inference
            ###################### Preparing inputs #########################
            with torch.no_grad():
                device = next(self.parameters()).device
                # # input_ids: [B, max_len] int ids, where `49407` for padding
                # # attention_mask: [B, max_len] {0, 1}. where `1` for valid, `0` for padding mask
                # batch_encoding = self.text_tokenizer(
                #     text=condition,
                #     return_attention_mask=True,
                #     padding=True,
                #     truncation=True,
                #     return_tensors="pt",
                # )

                # input_ids = batch_encoding.input_ids.to(device)
                # attention_mask = batch_encoding.attention_mask.to(device)

                # # last_hidden_state: [B, max_len, D]
                # # pooler_output: [B, D]
                # text_embedding: (
                #     transformers.modeling_outputs.BaseModelOutputWithPooling
                # ) = self.text_encoder(
                #     input_ids, attention_mask=attention_mask, return_dict=True
                # )
                # compressor.Codebooks: [4096, 32]

                # get class embedding from class_id to embedding
                class_embed = self.class_token[condition]
                # given shape and condition, produce token with secified shape
                # ================= start loop =================
                first_level_token = self.next_residual_predictor(None, class_embed)
                first_scale_feat = self.compressor.residual_forward(
                    first_level_token.unsqueeze(1), None, 0
                )

                predictions = [first_level_token]
                input_feats = [first_scale_feat]
                former_level_feat = first_scale_feat.copy()

                for i in range(1, len(self.compressor.Codebooks)):
                    next_level_token = self.next_residual_predictor(
                        (input_feats, i), class_embed
                    )
                    next_scale_feat = self.compressor.residual_forward(
                        next_level_token.unsqueeze(1), former_level_feat, i
                    )
                    former_level_feat = next_scale_feat.copy()
                    predictions.append(next_level_token)
                    transformer_input.append(next_scale_feat)

                # # list of [bs, hi, wi]
                # predictions.insert(0, first_level)
                # # list of [bs, 1, hi, wi]
                # predictions = [p.unsqueeze(1) for p in predictions]
                restored = self.compressor.decode(predictions)

                return predictions, restored


#################################################################################
#                            Core Transformer Model                             #
#################################################################################


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        qk_norm: bool,
    ):
        """
        Initialize the Attention module.

        Args:
            dim (int): Number of input dimensions.
            n_heads (int): Number of heads.
            n_kv_heads (Optional[int]): Number of kv heads, if using GQA.

        """
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(
            dim,
            n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * self.head_dim,
            dim,
            bias=False,
        )

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
            self.k_norm = nn.LayerNorm(self.n_kv_heads * self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

        # for proportional attention computation
        self.base_seqlen = None
        self.proportional_attn = False

    # @staticmethod
    # def apply_rotary_emb(
    #     x_in: torch.Tensor,
    #     freqs_cis: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     Apply rotary embeddings to input tensors using the given frequency
    #     tensor.

    #     This function applies rotary embeddings to the given query 'xq' and
    #     key 'xk' tensors using the provided frequency tensor 'freqs_cis'. The
    #     input tensors are reshaped as complex numbers, and the frequency tensor
    #     is reshaped for broadcasting compatibility. The resulting tensors
    #     contain rotary embeddings and are returned as real tensors.

    #     Args:
    #         x_in (torch.Tensor): Query or Key tensor to apply rotary embeddings.
    #         freqs_cis (torch.Tensor): Precomputed frequency tensor for complex
    #             exponentials.

    #     Returns:
    #         Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor
    #             and key tensor with rotary embeddings.
    #     """
    #     with torch.cuda.amp.autocast(enabled=False):
    #         x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
    #         freqs_cis = freqs_cis.unsqueeze(2)
    #         x_out = torch.view_as_real(x * freqs_cis).flatten(3)
    #         return x_out.type_as(x_in)

    # copied from huggingface modeling_llama.py
    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):

        def _get_unpad_data(attention_mask):
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            return (
                indices,
                cu_seqlens,
                max_seqlen_in_batch,
            )

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.n_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        """

        Args:
            x:
            y_feat:
            y_mask:
            pos_embed:

        Returns:

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

        pos_embed = pos_embed.reshape(bsz, seqlen, self.n_heads, self.head_dim)

        # xq = Attention.apply_rotary_emb(xq, freqs_cis=pos_embed)
        # xk = Attention.apply_rotary_emb(xk, freqs_cis=pos_embed)
        xq = xq + pos_embed
        xk = xk + pos_embed

        xq, xk = xq.to(dtype), xk.to(dtype)

        # if dtype in [torch.float16, torch.bfloat16]:
        #     # begin var_len flash attn
        #     (
        #         query_states,
        #         key_states,
        #         value_states,
        #         indices_q,
        #         cu_seq_lens,
        #         max_seq_lens,
        #     ) = self._upad_input(xq, xk, xv, x_mask, seqlen)

        #     cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        #     max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        #     if self.proportional_attn:
        #         softmax_scale = math.sqrt(
        #             math.log(seqlen, self.base_seqlen) / self.head_dim
        #         )
        #     else:
        #         softmax_scale = math.sqrt(1 / self.head_dim)

        #     attn_output_unpad = flash_attn_varlen_func(
        #         query_states,
        #         key_states,
        #         value_states,
        #         cu_seqlens_q=cu_seqlens_q,
        #         cu_seqlens_k=cu_seqlens_k,
        #         max_seqlen_q=max_seqlen_in_batch_q,
        #         max_seqlen_k=max_seqlen_in_batch_k,
        #         dropout_p=0.0,
        #         causal=False,
        #         softmax_scale=softmax_scale,
        #     )
        #     output = pad_input(attn_output_unpad, indices_q, bsz, seqlen)
        #     # end var_len_flash_attn

        # else:
        output = (
            F.scaled_dot_product_attention(
                xq.permute(0, 2, 1, 3),
                xk.permute(0, 2, 1, 3),
                xv.permute(0, 2, 1, 3),
                attn_mask=x_mask.expand(bsz, 1, -1, -1),
            )
            .permute(0, 2, 1, 3)
            .to(dtype)
        )

        output = output.flatten(-2)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple
                of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden
                dimension. Defaults to None.

        Attributes:
            w1 (Linear): Linear transformation for the first
                layer.
            w2 (Linear): Linear transformation for the second layer.
            w3 (Linear): Linear transformation for the third
                layer.

        """
        super().__init__()
        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

    # @torch.compile
    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class ProjLayer(nn.Module):
    """
    The prjection layer of Var
    Upsample hidden_size -> 4 * hidden_size
    """

    def __init__(self, hidden_size, scale_factor=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.proj = nn.Linear(hidden_size, scale_factor * hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        # [PxP, 4 * hidden_size]
        x = self.proj(x)

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, prediction_num):
        super().__init__()
        # if prototypes is not None:
        #     self.register_buffer("prototypes", prototypes)
        #     self.prototype_proj = nn.Linear(prototypes.shape[-1], hidden_size)
        #     self.norm_final = nn.InstanceNorm2d(hidden_size, affine=False, eps=1e-6)
        #     self.linear = nn.Conv2d(hidden_size, hidden_size, 1)
        #     self.skip_connection = nn.Conv2d(hidden_size, hidden_size, 1)
        #     self.adaLN_modulation = nn.Sequential(
        #         nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        #     )

        # else:
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, prediction_num)
        # self.skip_connection = nn.Linear(hidden_size, prediction_num)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )

        # Zero-out adaLN modulation layers in DiT blocks:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, condition):
        # if hasattr(self, "prototypes"):
        #     # [B, D]
        #     shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        #     # modulate, [B, D, H, W]
        #     out = (
        #         self.norm_final(x) * (1 + scale[:, :, None, None])
        #         + shift[:, :, None, None]
        #     )
        #     # [B, D, H, W]
        #     x = self.linear(out) + self.skip_connection(x)
        #     n, c, h, w = x.shape
        #     # [K, D]
        #     codes = self.prototype_proj(self.prototypes)
        #     # [b, hw, d] @ [b, d, k] -> [b, hw, k]
        #     similarity = torch.bmm(
        #         x.permute(0, 2, 3, 1).reshape(n, h * w, c).contiguous(),
        #         codes.expand(n, *codes.shape).permute(0, 2, 1).contiguous(),
        #     )
        #     # [b, k, h, w]
        #     similarity = (
        #         similarity.reshape(n, h, w, -1).permute(0, 3, 1, 2).contiguous()
        #     )
        #     return similarity
        # else:
        # [B, D]
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        # modulate, [B, L, D]
        out = self.norm_final(x) * (1 + scale[:, None]) + shift[:, None]
        x = self.linear(out)  # + self.skip_connection(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
    ) -> None:
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            dim (int): Embedding dimension of the input features.
            n_heads (int): Number of attention heads.
            n_kv_heads (Optional[int]): Number of attention heads in key and
                value features (if using GQA), or set to None for the same as
                query.
            multiple_of (int):
            ffn_dim_multiplier (float):
            norm_eps (float):

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm)
        self.ffn = FeedForward(dim=dim, hidden_dim=4 * dim)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                dim,
                6 * dim,
                bias=True,
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize transformer layers and proj layer:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.zeros_(self.adaLN_modulation[1].weight)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        y_emb: torch.Tensor,
        pos_embed: torch.Tensor,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and
                feedforward layers.

        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(y_emb).chunk(6, dim=1)
        )
        x = x + self.attention_norm1(
            gate_msa.unsqueeze(1)
            * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa),
                x_mask,
                pos_embed,
            )
        )
        d = x.shape[-1]
        x = x + self.ffn_norm1(
            gate_mlp.unsqueeze(1)
            * self.ffn(
                modulate(self.ffn_norm(x), shift_mlp, scale_mlp).reshape(-1, d),
            ).reshape(*x.shape)
        )

        return x


class Transformer(nn.Module):
    """
    Next scale model with a Transformer backbone.
    """

    def __init__(
        self,
        input_dim,
        canvas_size,  # e.g.: 32, corresponding to raw pixels: 512
        cap_dim,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        vocab_size=4096,
        norm_eps=1e-5,
        qk_norm=False,
        # learn_sigma=True,
    ):
        super().__init__()
        # self.learn_sigma = learn_sigma
        self.in_channels = input_dim
        self.canvas_size = canvas_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # self.input_transform = nn.Sequential(
        #     nn.InstanceNorm2d(input_dim, eps=1e-6),
        #     nn.Conv2d(input_dim, hidden_size, 1, bias=True),
        # )

        self.final_layer = checkpoint_wrapper(FinalLayer(hidden_size, vocab_size))
        self.cap_embedder = nn.Sequential(
            nn.LayerNorm(cap_dim),
            nn.Linear(
                cap_dim,
                hidden_size,
                bias=True,
            ),
        )
        self.token_embedder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(
                input_dim,
                hidden_size,
                bias=True,
            ),
        )

        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.num_patches = (
            canvas_size * canvas_size * 64
        )  # expand canvas to 8 * canvas, to support at least 8*resolution
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.num_patches, hidden_size),
                std=2 / (5 * self.hidden_size),
            ),
            requires_grad=False,
        )

        self.blocks = nn.ModuleList(
            [
                checkpoint_wrapper(
                    TransformerBlock(
                        idx, hidden_size, num_heads, num_heads, norm_eps, qk_norm
                    )
                )
                for idx in range(depth)
            ]
        )
        self.proj_layer = checkpoint_wrapper(ProjLayer(hidden_size, scale_factor=1))

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize transformer layers and proj layer:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # if module.bias is not None:
                #     nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.zeros_(self.cap_embedder[1].weight)

    def random_pos_embed(self, bs, h, w):
        H = W = int(math.sqrt(self.num_patches))
        # [H, W, D]
        pos_embed = self.pos_embed.reshape(H, W, -1)
        result = list()
        for i in range(bs):
            random_up = random.randint(0, H - h)
            random_left = random.randint(0, W - w)
            result.append(
                pos_embed[
                    random_up : random_up + h, random_left : random_left + w
                ].reshape(h * w, -1)
            )
        # [bs, hw, d]
        return torch.stack(result)

    def center_pos_embed(self, h, w):
        H = W = int(math.sqrt(self.num_patches))
        # [H, W, D]
        pos_embed = self.pos_embed.reshape(H, W, -1)
        up_start = (H - h) // 2
        left_start = (W - w) // 2
        return pos_embed[up_start : up_start + h, left_start : left_start + w].reshape(
            h * w, -1
        )

    def unpatchify(self, x, h, w):
        """
        x: (bs, patch_size**2, 4 * D)
        imgs: (bs, H, W, D)
        """
        bs, hw, dim = x.shape

        return (
            x.permute(0, 2, 1).reshape(bs, dim, h, w).contiguous()
        )  # [bs, 4 * D, h, w]
        return self.pixel_shuffle(x)  # [bs, D, 2 * h, 2 * w]

    def forward(self, x, x_mask, cap_pooled):
        # 0, 1, 2, 3
        # bs, c, h, w = x.shape
        # # [b, h, w, d]
        # # x = self.x[code]
        # # [b, h*w, hidden]
        # x = (
        #     self.input_transform(x)
        #     .permute(0, 2, 3, 1)
        #     .reshape(bs, h * w, -1)
        #     .contiguous()
        # )

        bs = len(x)
        x = self.token_embedder(x)
        # if self.training:
        #     # TODO: change to random
        #     selected_pos_embed = self.random_pos_embed(bs, h, w)
        #     # selected_pos_embed = self.center_pos_embed(h, w)
        # else:
        #     selected_pos_embed = self.center_pos_embed(h, w)

        selected_pos_embed = self.pos_embed[: x.shape[1]].expand(bs, x.shape[1], -1)
        cap_emb = self.cap_embedder(cap_pooled)

        for block in self.blocks:
            x = block(
                x,
                x_mask,
                cap_emb,
                selected_pos_embed,
            )
            # print(x.mean(), x.std())
        # x = self.unpatchify(x, h ,w) # [bs, hidden, 2h, 2w]
        # x = (
        #     x.reshape(bs, h, w, -1).permute(0, 3, 1, 2).contiguous()
        # )  # [bs, hidden, h, w]
        prediction = self.final_layer(x, cap_emb)  # [bs, k, 2h, 2w]
        # print(prediction[0, :, 0, 0].min(), prediction[0, :, 0, 0].max(), prediction[0, :, 0, 0].softmax(-1).min(), prediction[0, :, 0, 0].softmax(-1).max())
        return prediction

    # def forward_with_cfg(self, x, t, y, cfg_scale):
    #     """
    #     Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
    #     """
    #     # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    #     half = x[: len(x) // 2]
    #     combined = torch.cat([half, half], dim=0)
    #     model_out = self.forward(combined, t, y)
    #     # For exact reproducibility reasons, we apply classifier-free guidance on only
    #     # three channels by default. The standard approach to cfg applies it to all channels.
    #     # This can be done by uncommenting the following line and commenting-out the line following that.
    #     # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
    #     eps, rest = model_out[:, :3], model_out[:, 3:]
    #     cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    #     half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    #     eps = torch.cat([half_eps, half_eps], dim=0)
    #     return torch.cat([eps, rest], dim=1)


class AnyResolutionModel(nn.Module):
    def __init__(
        self,
        # the token should be firstly aligned to cap_dim, then lift to hidden_size
        # cap_dim,
        # NOTE: from low resolution to high resolution. this list should be ascending.
        canvas_size: List[int],  # e.g.: 32, corresponding to raw pixels: 512
        codebooks,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        qk_norm=False,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.token_dim = codebooks[0][-1]
        self.hidden_size = hidden_size
        # NOTE: we only need first (level - 1) blocks
        self.model = Transformer(
            codebooks[0][-1],
            canvas_size[-1],
            hidden_size,
            hidden_size,
            depth,
            num_heads,
            codebooks[0][0],
            norm_eps,
            qk_norm,
        )
        # self.text_lift = nn.ModuleList(
        #     [nn.Linear(cap_dim, hidden_size) for _ in codebooks]
        # )

        self.first_level_pos_embed = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(1, canvas_size[-1] * canvas_size[-1], self.token_dim),
                std=math.sqrt(2 / (5 * self.token_dim)),
            )
        )
        self.cap_to_first_token = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(
                hidden_size,
                self.token_dim,
                bias=True,
            ),
        )

        self.level_indicator_pos_embed = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(len(canvas_size), self.token_dim),
                std=math.sqrt(2 / (5 * self.token_dim)),
            )
        )

        self.register_buffer("input_mask", self.prepare_input_mask(canvas_size), False)
        # self.blocks = nn.ModuleList([AnyResolutionBlock(canvas_size[-1], in_channels, hidden_size, depth, num_heads, mlp_ratio) for _ in canvas_size] * len(canvas_size))

    def prepare_input_mask(self, canvas_size):
        lengths = list()
        for c in canvas_size:
            lengths.append(c * c)
        # [L, L]
        attention_mask = torch.tril(torch.ones([sum(lengths), sum(lengths)]))
        curDiag = 0
        for l in lengths:
            attention_mask[curDiag : curDiag + l, curDiag : curDiag + l] = 1
            curDiag += l
        return attention_mask

    def random_pos_embed(self, bs, h, w):
        H = W = int(math.sqrt(self.first_level_pos_embed.shape[1]))
        # [H, W, D]
        pos_embed = self.first_level_pos_embed.reshape(H, W, -1)
        result = list()
        for i in range(bs):
            random_up = random.randint(0, H - h)
            random_left = random.randint(0, W - w)
            result.append(
                pos_embed[
                    random_up : random_up + h, random_left : random_left + w
                ].reshape(h * w, -1)
            )
        # [bs, hw, d]
        return torch.stack(result)

    def center_pos_embed(self, h, w):
        H = W = int(math.sqrt(self.first_level_pos_embed.shape[1]))
        # [H, W, D]
        pos_embed = self.first_level_pos_embed.reshape(H, W, -1)
        up_start = (H - h) // 2
        left_start = (W - w) // 2
        return pos_embed[up_start : up_start + h, left_start : left_start + w].reshape(
            h * w, -1
        )

    def forward(self, all_forwards_for_residual, cap_pooled):
        if self.training:
            if not isinstance(all_forwards_for_residual, list):
                raise RuntimeError("The given training input is not a list.")
            results = list()
            total = list()
            for level, current in enumerate(all_forwards_for_residual):
                if level == 0:
                    if current is not None:
                        raise RuntimeError("The first level input should be None.")
                    bs, _, h, w = all_forwards_for_residual[1].shape
                    h, w = 1, 1
                    # current should be picked from pos_embed
                    if self.training:
                        # TODO: change to random
                        # selected_pos_embed = self.random_pos_embed(bs, h, w)
                        selected_pos_embed = self.center_pos_embed(h, w)
                    else:
                        selected_pos_embed = self.center_pos_embed(h, w)
                    current = selected_pos_embed.expand(
                        bs, h * w, self.token_dim
                    )  # [bs, hw, hidden]
                    current = (
                        selected_pos_embed
                        + self.cap_to_first_token(cap_pooled)[:, None, ...]
                    )
                    current = (
                        current.permute(0, 2, 1)
                        .reshape(bs, self.token_dim, h, w)
                        .contiguous()
                    )

                level_emb = self.level_indicator_pos_embed[level]
                bs, _, h, w = current.shape
                # current = torch.cat([level_emb, current], dim=1) # todo: maybe useful
                # [bs, hw, c]
                current = current.permute(0, 2, 3, 1).reshape(bs, h * w, -1) + level_emb
                # [bs, hw, c]
                total.append(current)
            # [bs, h1w1+h2w2+h3w3..., c]
            total = torch.cat(total, dim=1)
            # [bs, h1w1+h2w2+h3w3.., k]
            results = self.model(total, self.input_mask, cap_pooled)
            # NOTE: in training, the len of reuslts is level - 1
            return results
        else:
            # inference
            if all_forwards_for_residual is None:
                # shape = [int(x) for x in all_forwards_for_residual.split(",")]
                # bs, h, w = shape
                h, w = 1, 1
                bs, dim = cap_poooled.shape
                selected_pos_embed = self.center_pos_embed(h, w)
                current = selected_pos_embed.expand(
                    bs, h * w, self.token_dim
                )  # [bs, hw, hidden]
                current = current + self.cap_to_first_token(cap_pooled)[:, None, ...]
                current = (
                    current.permute(0, 2, 1)
                    .reshape(bs, self.token_dim, h, w)
                    .contiguous()
                )
                level_emb = self.level_indicator_pos_embed[0]
                current = current + level_emb[:, None, None]
                predict = self.model(current, self.input_mask, cap_pooled)
                
                return predict.argmax(dim=1)
            else:
                # [bs, c, h*2, w*2]
                current, level = all_forwards_for_residual
                level_emb = self.level_indicator_pos_embed[level]
                # current = torch.cat([level_emb, current], dim=1) # todo: maybe useful
                current = current + level_emb[:, None, None]
                predict = self.model(current, self.input_mask, cap_pooled)
                # NOTE: in training, the len of reuslts is level - 1
                
                return predict.argmax(dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                 AnyRes Configs                                #
#################################################################################
# 2.20B
def AnyRes_XL(canvas_size, codebooks, **kwargs):
    return AnyResolutionModel(
        canvas_size,
        codebooks[1:],
        depth=28,
        hidden_size=2304,
        num_heads=16,
        **kwargs,
    )


# 1.51B
def AnyRes_L(canvas_size, codebooks, **kwargs):
    return AnyResolutionModel(
        canvas_size,
        codebooks,
        depth=24,
        hidden_size=1536,
        num_heads=16,
        **kwargs,
    )


# 480M
def AnyRes_B(canvas_size, codebooks, **kwargs):
    return AnyResolutionModel(
        canvas_size,
        codebooks[1:],
        depth=20,
        hidden_size=1152,
        num_heads=16,
        **kwargs,
    )


# 152.1455M
def AnyRes_S(canvas_size, codebooks, **kwargs):
    return AnyResolutionModel(
        canvas_size,
        codebooks[1:],
        depth=16,
        hidden_size=768,
        num_heads=16,
        **kwargs,
    )


# AnyRes_models = {
#     'AnyRes-XL/2': AnyRes_XL_2,  'AnyRes-XL/4': AnyRes_XL_4,  'AnyRes-XL/8': AnyRes_XL_8,
#     'AnyRes-L/2':  AnyRes_L_2,   'AnyRes-L/4':  AnyRes_L_4,   'AnyRes-L/8':  AnyRes_L_8,  "AnyRes-L/16": AnyRes_L_16, "AnyRes-L/32": AnyRes_L_32,
#     'AnyRes-B/2':  AnyRes_B_2,   'AnyRes-B/4':  AnyRes_B_4,   'AnyRes-B/8':  AnyRes_B_8,
#     'AnyRes-S/2':  AnyRes_S_2,   'AnyRes-S/4':  AnyRes_S_4,   'AnyRes-S/8':  AnyRes_S_8,
# }
