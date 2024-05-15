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
class GeneratorV3(nn.Module):
    def __init__(
        self,
        channel: int,
        k: List[int],
        denseNorm: bool,
        loadFrom: str,
        qk_norm: bool,
        norm_eps: float,
        *_,
        **__,
    ):
        super().__init__()
        self.compressor = Neon(channel, k, denseNorm)
        state_dict = torch.load(loadFrom, map_location="cpu")
        self.compressor.load_state_dict(
            {
                k[len("module._compressor.") :]: v
                for k, v in state_dict["trainer"]["_model"].items()
            }
        )
        for params in self.compressor.parameters():
            params.requires_grad_(False)
        logging.info("Loaded compressor checkpoint from %s.", loadFrom)

        logging.debug("Start loading clip...")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True
        )
        logging.debug("Loaded clip text model from %s.", "openai/clip-vit-base-patch32")
        self.text_tokenizer = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True
        )
        logging.debug("Loaded clip text model from %s.", "openai/clip-vit-base-patch32")
        for params in self.text_encoder.parameters():
            params.requires_grad_(False)

        clip_text_channels = self.text_encoder.text_model.config.hidden_size

        # NOTE: text_to_first_level: This transforms text embeddings to the first level token
        # NOTE: next_residual_predictor: we only need first (level - 1) codebook, and corresponding canvas.
        # NOTE: remove first dim of codebook, since it is for product quantization
        self.next_residual_predictor = AnyRes_XL(
            clip_text_channels,
            [2, 4, 8, 16],
            [codebook.squeeze(0) for codebook in self.compressor.Codebooks],
            qk_norm=qk_norm,
            norm_eps=norm_eps,
        )
        logging.debug("Created any-res transformer.")

        self.compressor.eval()
        self.text_encoder.eval()
        # Cast to bfloat16
        # self.text_encoder.float16()
        # self.text_to_first_level.bfloat16()
        # self.next_residual_predictor.bfloat16()

    def train(self, mode: bool = True):
        self.compressor.eval()
        self.text_encoder.eval()
        return super().train(mode)

    def forward(self, image, condition: List[str]):
        if not isinstance(condition, list):
            raise NotImplementedError
        if self.training:
            ###################### Preparing inputs #########################
            with torch.no_grad():
                # list of [n, 1, h, w], len of list == levels
                # from low resolution to high resolution
                # NOTE: for reflection padding, the input tensor size (`.numel()`) should not exceed 2^32
                # NOTE: therefore, we manually split image into batch 16
                splitted = torch.split(image, 16)
                allCodes = list()
                all_forwards_for_residual = list()
                for sp in splitted:
                    codes = self.compressor.encode(sp)
                    allCodes.append(codes)
                    this_split_forward_residual = list()
                    formerLevel = None
                    for level, code in enumerate(codes[:-1]):
                        this_split_forward_residual.append(
                            self.compressor.residual_forward(code, formerLevel, level)
                        )
                        formerLevel = this_split_forward_residual[-1]
                    all_forwards_for_residual.append(this_split_forward_residual)
                codes = [torch.cat(x) for x in zip(*allCodes)]
                # list - 1 of [n, c, 2h, 2w]
                all_forwards_for_residual = [
                    torch.cat(x) for x in zip(*all_forwards_for_residual)
                ]

                # input_ids: [B, max_len] int ids, where `49407` for padding
                # attention_mask: [B, max_len] {0, 1}. where `1` for valid, `0` for padding mask
                batch_encoding = self.text_tokenizer(
                    text=condition,
                    return_attention_mask=True,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

                input_ids = batch_encoding.input_ids.to(image.device)
                attention_mask = batch_encoding.attention_mask.to(image.device)

                # last_hidden_state: [B, max_len, D]
                # pooler_output: [B, D]
                text_embedding: (
                    transformers.modeling_outputs.BaseModelOutputWithPooling
                ) = self.text_encoder(
                    input_ids, attention_mask=attention_mask, return_dict=True
                )

            # NOTE: remove product quantization artifacts, since we don't use product quantization
            codes = [c.squeeze(1) for c in codes]

            # given shape and condition, produce token with secified shape
            # first_level = self.text_to_first_level(
            #     codes[0].shape,
            #     text_embedding.pooler_output.detach().clone(),
            #     text_embedding.last_hidden_state.detach().clone(),
            #     attention_mask,
            # )

            predictions = self.next_residual_predictor(
                [None, *all_forwards_for_residual],
                text_embedding.pooler_output.detach().clone(),
                text_embedding.last_hidden_state.detach().clone(),
                attention_mask.bool(),
            )

            loss = [
                F.cross_entropy(pre, gt, reduction="none")
                for pre, gt in zip([*predictions], codes)
            ]

            # list of [n, 1, h, w], len of list == levels
            restoredCodes = [
                pre.detach().clone().argmax(1, keepdim=True) for pre in predictions
            ]
            # [n, 1, h, w]
            # restoredCodes.insert(
            #     0, first_level.detach().clone().argmax(1, keepdim=True)
            # )
            with torch.no_grad():
                splitted = list(zip(*list(torch.split(x, 16) for x in restoredCodes)))

                allRestored = list()
                for sp in splitted:
                    allRestored.append(self.compressor.decode(sp))
                restored = torch.cat(allRestored)

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
            ###################### Preparing inputs #########################
            with torch.no_grad():
                device = next(self.parameters()).device
                # input_ids: [B, max_len] int ids, where `49407` for padding
                # attention_mask: [B, max_len] {0, 1}. where `1` for valid, `0` for padding mask
                batch_encoding = self.text_tokenizer(
                    text=condition,
                    return_attention_mask=True,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

                input_ids = batch_encoding.input_ids.to(device)
                attention_mask = batch_encoding.attention_mask.to(device)

                # last_hidden_state: [B, max_len, D]
                # pooler_output: [B, D]
                text_embedding: (
                    transformers.modeling_outputs.BaseModelOutputWithPooling
                ) = self.text_encoder(
                    input_ids, attention_mask=attention_mask, return_dict=True
                )

                # given shape and condition, produce token with secified shape
                first_level = self.next_residual_predictor(
                    f"{len(text_embedding)},2,2",
                    text_embedding.pooler_output.detach().clone(),
                    text_embedding.last_hidden_state.detach().clone(),
                    attention_mask.bool(),
                )

                formerLevel = self.compressor.residual_forward(
                    first_level.unsqueeze(1), None, 0
                )

                predictions = list()
                for i in range(0, len(self.compressor.Codebooks) - 1):
                    predictions.append(
                        self.next_residual_predictor(
                            (formerLevel, i),
                            text_embedding.pooler_output.detach().clone(),
                            text_embedding.last_hidden_state.detach().clone(),
                            attention_mask.bool(),
                        )
                    )
                    formerLevel = self.compressor.residual_forward(
                        predictions[-1].unsqueeze(1), formerLevel, i
                    )

                # list of [bs, hi, wi]
                predictions.insert(0, first_level)
                # list of [bs, 1, hi, wi]
                predictions = [p.unsqueeze(1) for p in predictions]

                splitted = list(zip(*list(torch.split(x, 16) for x in predictions)))

                allRestored = list()
                for sp in splitted:
                    allRestored.append(self.compressor.decode(sp))
                restored = torch.cat(allRestored)
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
        y_dim: int,
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
        if y_dim > 0:
            self.wk_y = nn.Linear(
                y_dim,
                self.n_kv_heads * self.head_dim,
                bias=False,
            )
            self.wv_y = nn.Linear(
                y_dim,
                self.n_kv_heads * self.head_dim,
                bias=False,
            )
            self.gate = nn.Parameter(torch.zeros([self.n_heads]))

        self.wo = nn.Linear(
            n_heads * self.head_dim,
            dim,
            bias=False,
        )

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
            self.k_norm = nn.LayerNorm(self.n_kv_heads * self.head_dim)
            if y_dim > 0:
                self.ky_norm = nn.LayerNorm(self.n_kv_heads * self.head_dim)
            else:
                self.ky_norm = nn.Identity()
        else:
            self.q_norm = self.k_norm = nn.Identity()
            self.ky_norm = nn.Identity()

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
        y_feat: torch.Tensor,
        y_mask: torch.Tensor,
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
            )
            .permute(0, 2, 1, 3)
            .to(dtype)
        )

        if hasattr(self, "wk_y"):
            # todo better flash_attn support
            yk = self.ky_norm(self.wk_y(y_feat)).reshape(
                bsz, -1, self.n_kv_heads, self.head_dim
            )
            yv = self.wv_y(y_feat).reshape(bsz, -1, self.n_kv_heads, self.head_dim)
            n_rep = self.n_heads // self.n_kv_heads

            if n_rep >= 1:
                yk = yk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                yv = yv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

            output_y = F.scaled_dot_product_attention(
                xq.permute(0, 2, 1, 3),
                yk.permute(0, 2, 1, 3),
                yv.permute(0, 2, 1, 3),
                y_mask.reshape(bsz, 1, 1, -1).expand(bsz, self.n_heads, seqlen, -1),
            ).permute(0, 2, 1, 3)

            output_y = output_y * self.gate.tanh().reshape(1, 1, -1, 1)
            output = output + output_y

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

    def __init__(self, hidden_size, prediction_num, prototypes=None):
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
        self.norm_final = nn.InstanceNorm2d(hidden_size, affine=False, eps=1e-6)
        self.linear = nn.Conv2d(hidden_size, prediction_num, 1)
        self.skip_connection = nn.Conv2d(hidden_size, prediction_num, 1)
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
        # modulate, [B, D, H, W]
        out = (
            self.norm_final(x) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        )
        x = self.linear(out) + self.skip_connection(x)
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
        y_dim: int,
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
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm, y_dim)
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

        self.attention_y_norm = RMSNorm(y_dim, eps=norm_eps)

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
        y_emb: torch.Tensor,
        y_feat: torch.Tensor,
        y_mask: torch.Tensor,
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
                self.attention_y_norm(y_feat),
                y_mask,
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
        cap_dim=768,
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

        self.input_transform = nn.Sequential(
            nn.InstanceNorm2d(input_dim, eps=1e-6),
            nn.Conv2d(input_dim, hidden_size, 1, bias=True),
        )

        self.final_layer = checkpoint_wrapper(FinalLayer(hidden_size, vocab_size, None))
        self.cap_embedder = nn.Sequential(
            nn.LayerNorm(cap_dim),
            nn.Linear(
                cap_dim,
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
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                checkpoint_wrapper(
                    TransformerBlock(
                        idx, hidden_size, num_heads, 1, norm_eps, qk_norm, cap_dim
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

    def random_pos_embed(self, h, w):
        H = W = int(math.sqrt(self.num_patches))
        # [H, W, D]
        pos_embed = self.pos_embed.reshape(H, W, -1)
        random_up = random.randint(0, H - h - 1)
        random_left = random.randint(0, W - w - 1)
        return pos_embed[
            random_up : random_up + h, random_left : random_left + w
        ].reshape(h * w, -1)

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

    def forward(self, x, cap_pooled, cap_cond, cap_mask):
        # 0, 1, 2, 3
        bs, c, h, w = x.shape
        # [b, h, w, d]
        # x = self.x[code]
        # [b, h*w, hidden]
        x = (
            self.input_transform(x)
            .permute(0, 2, 3, 1)
            .reshape(bs, h * w, -1)
            .contiguous()
        )

        if self.training:
            # TODO: change to random
            # selected_pos_embed = self.random_pos_embed(h, w)
            selected_pos_embed = self.center_pos_embed(h, w)
        else:
            selected_pos_embed = self.center_pos_embed(h, w)

        cap_emb = self.cap_embedder(cap_pooled)
        # x = x + selected_pos_embed # [bs, hw, hidden]
        for block in self.blocks:
            x = block(
                x,
                cap_emb,
                cap_cond,
                cap_mask,
                selected_pos_embed.expand(bs, h * w, self.hidden_size),
            )
            # print(x.mean(), x.std())
        # x = self.unpatchify(x, h ,w) # [bs, hidden, 2h, 2w]
        x = (
            x.reshape(bs, h, w, -1).permute(0, 3, 1, 2).contiguous()
        )  # [bs, hidden, h, w]
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
        cap_dim,
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
        self.cap_dim = cap_dim
        self.token_dim = codebooks[0].shape[-1]
        self.hidden_size = hidden_size
        # NOTE: we only need first (level - 1) blocks
        self.model = Transformer(
            codebooks[0].shape[-1],
            canvas_size[-1],
            cap_dim,
            hidden_size,
            depth,
            num_heads,
            len(codebooks[0]),
            qk_norm,
            norm_eps,
        )
        # self.text_lift = nn.ModuleList(
        #     [nn.Linear(cap_dim, hidden_size) for _ in codebooks]
        # )

        self.first_level_pos_embed = nn.Parameter(
            nn.init.uniform_(
                torch.empty(1, canvas_size[-1] * canvas_size[-1], self.token_dim),
                a=-math.sqrt(2 / (5 * hidden_size)),
                b=math.sqrt(2 / (5 * hidden_size)),
            )
        )

        self.level_indicator_pos_embed = nn.Parameter(torch.randn([5, self.token_dim]))
        # self.blocks = nn.ModuleList([AnyResolutionBlock(canvas_size[-1], in_channels, hidden_size, depth, num_heads, mlp_ratio) for _ in canvas_size] * len(canvas_size))

    def random_pos_embed(self, h, w):
        H = W = int(math.sqrt(self.first_level_pos_embed.shape[1]))
        # [H, W, D]
        pos_embed = self.first_level_pos_embed.reshape(H, W, -1)
        random_up = random.randint(0, H - h - 1)
        random_left = random.randint(0, W - w - 1)
        return pos_embed[
            random_up : random_up + h, random_left : random_left + w
        ].reshape(h * w, -1)

    def center_pos_embed(self, h, w):
        H = W = int(math.sqrt(self.first_level_pos_embed.shape[1]))
        # [H, W, D]
        pos_embed = self.first_level_pos_embed.reshape(H, W, -1)
        up_start = (H - h) // 2
        left_start = (W - w) // 2
        return pos_embed[up_start : up_start + h, left_start : left_start + w].reshape(
            h * w, -1
        )

    def forward(self, all_forwards_for_residual, cap_pooled, cap_cond, cap_mask):
        if self.training:
            if not isinstance(all_forwards_for_residual, list):
                raise RuntimeError("The given training input is not a list.")
            results = list()
            for level, current in enumerate(all_forwards_for_residual):
                if level == 0:
                    if current is not None:
                        raise RuntimeError("The first level input should be None.")
                    bs, _, h, w = all_forwards_for_residual[1].shape
                    h = h // 2
                    w = w // 2
                    # current should be picked from pos_embed
                    if self.training:
                        # TODO: change to random
                        # selected_pos_embed = self.random_pos_embed(h, w)
                        selected_pos_embed = self.center_pos_embed(h, w)
                    else:
                        selected_pos_embed = self.center_pos_embed(h, w)
                    current = selected_pos_embed.expand(
                        bs, h * w, self.token_dim
                    )  # [bs, hw, hidden]
                    current = (
                        current.permute(0, 2, 1)
                        .reshape(bs, self.token_dim, h, w)
                        .contiguous()
                    )

                level_emb = self.level_indicator_pos_embed[level]
                # current = torch.cat([level_emb, current], dim=1) # todo: maybe useful
                # [bs, c, h*2, w*2]
                current = current + level_emb[:, None, None]
                # [bs, k, h*2, w*2]
                results.append(self.model(current, cap_pooled, cap_cond, cap_mask))
            # NOTE: in training, the len of reuslts is level - 1
            return results
        else:
            if isinstance(all_forwards_for_residual, str):
                shape = [int(x) for x in all_forwards_for_residual.split(",")]
                bs, h, w = shape
                selected_pos_embed = self.center_pos_embed(h, w)
                current = selected_pos_embed.expand(
                    bs, h * w, self.token_dim
                )  # [bs, hw, hidden]
                current = (
                    current.permute(0, 2, 1)
                    .reshape(bs, self.token_dim, h, w)
                    .contiguous()
                )
                level_emb = self.level_indicator_pos_embed[0]
                current = current + level_emb[:, None, None]
                predict = self.model(current, cap_pooled, cap_cond, cap_mask)
                return predict.argmax(1)
            else:
                # [bs, c, h*2, w*2]
                current, level = all_forwards_for_residual
                level_emb = self.level_indicator_pos_embed[level]
                # current = torch.cat([level_emb, current], dim=1) # todo: maybe useful
                current = current + level_emb[:, None, None]
                predict = self.model(current, cap_pooled, cap_cond, cap_mask)
                # NOTE: in training, the len of reuslts is level - 1
                return predict.argmax(1)


# class CapConditionedModel(nn.Module):
#     def __init__(
#         self,
#         cap_dim: int,
#         codebook,
#         hidden_size=1152,
#         depth=28,
#         num_heads=16,
#         mlp_ratio=4.0,
#         qk_norm=False,
#         norm_eps=1e-5,
#     ):
#         super().__init__()
#         # self.learn_sigma = learn_sigma
#         self.cap_dim = cap_dim
#         self.canvas_size = 16  # 16 -> max resolution 4096
#         self.num_heads = num_heads
#         self.hidden_size = hidden_size

#         self.text_lift = nn.Linear(cap_dim, hidden_size, bias=False)
#         self.pre_layer = checkpoint_wrapper(FinalLayer(hidden_size, hidden_size))
#         self.post_layer = checkpoint_wrapper(ProjLayer(hidden_size, 1))

#         # we only need level - 1 final layers.
#         self.final_layer = checkpoint_wrapper(
#             FinalLayer(hidden_size, len(codebook), None)
#         )

#         self.num_patches = self.canvas_size * self.canvas_size
#         # For first level, we need to pick random trainable embedding for startup
#         # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=True)
#         self.pos_embed = nn.Parameter(
#             nn.init.uniform_(
#                 torch.empty(1, self.num_patches, hidden_size),
#                 a=-math.sqrt(2 / (5 * hidden_size)),
#                 b=math.sqrt(2 / (5 * hidden_size)),
#             )
#         )

#         self.blocks = nn.ModuleList(
#             [
#                 checkpoint_wrapper(Attention(hidden_size, num_heads, 1, qk_norm, 768))
#                 for _ in range(depth)
#             ]
#         )
#         self._initialize_weights()

#     def _initialize_weights(self):
#         # Initialize transformer layers and proj layer:
#         def _basic_init(module):
#             if isinstance(module, nn.Linear):
#                 torch.nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.constant_(module.bias, 0)

#         self.apply(_basic_init)

#         # nn.init.xavier_normal_(self.pos_embed)

#         # Initialize (and freeze) pos_embed by sin-cos embedding:
#         # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
#         # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

#         # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
#         # w = self.x_embedder.proj.weight.data
#         # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#         # nn.init.constant_(self.x_embedder.proj.bias, 0)

#         # Zero-out adaLN modulation layers in DiT blocks:
#         # for block in self.blocks:
#         # nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
#         # nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

#         # Zero-out output layers:
#         # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
#         # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
#         # nn.init.constant_(self.final_layer.linear.weight, 0)
#         # nn.init.constant_(self.final_layer.linear.bias, 0)

#     def random_pos_embed(self, h, w):
#         H = W = int(math.sqrt(self.num_patches))
#         # [H, W, D]
#         pos_embed = self.pos_embed.reshape(H, W, -1)
#         random_up = random.randint(0, H - h - 1)
#         random_left = random.randint(0, W - w - 1)
#         return pos_embed[
#             random_up : random_up + h, random_left : random_left + w
#         ].reshape(h * w, -1)

#     def center_pos_embed(self, h, w):
#         H = W = int(math.sqrt(self.num_patches))
#         # [H, W, D]
#         pos_embed = self.pos_embed.reshape(H, W, -1)
#         up_start = (H - h) // 2
#         left_start = (W - w) // 2
#         return pos_embed[up_start : up_start + h, left_start : left_start + w].reshape(
#             h * w, -1
#         )

#     def forward(
#         self, target_shape, pooled_condition, sequence_condition, attention_mask
#     ):
#         # 0, 1, 2
#         bs, h, w = target_shape

#         pooled_condition = self.text_lift(pooled_condition)
#         sequence_condition = self.text_lift(sequence_condition)

#         if self.training:
#             # TODO: change to random
#             # selected_pos_embed = self.random_pos_embed(h, w)
#             selected_pos_embed = self.center_pos_embed(h, w)
#         else:
#             selected_pos_embed = self.center_pos_embed(h, w)

#         x = selected_pos_embed.expand(bs, h * w, self.hidden_size)  # [bs, hw, hidden]
#         x = (
#             self.pre_layer(
#                 x.permute(0, 2, 1).reshape(bs, self.hidden_size, h, w).contiguous(),
#                 pooled_condition,
#             )
#             .permute(0, 2, 3, 1)
#             .reshape(bs, h * w, self.hidden_size)
#             .contiguous()
#         )
#         for block, cross in zip(self.blocks):
#             x = block(x, pooled_condition)
#         x = self.post_layer(x)
#         x = x.reshape(bs, h, w, -1).permute(0, 3, 1, 2).contiguous()
#         # x = x.permute(0, )
#         prediction = self.final_layer(x, pooled_condition)  # [bs, k, h, w]
#         # print(prediction[0, :, 0, 0].min(), prediction[0, :, 0, 0].max(), prediction[0, :, 0, 0].softmax(-1).min(), prediction[0, :, 0, 0].softmax(-1).max())
#         return prediction if self.training else prediction.argmax(1)


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
def AnyRes_XL(cap_dim, canvas_size, codebooks, **kwargs):
    return AnyResolutionModel(
        cap_dim,
        canvas_size,
        codebooks[1:],
        depth=28,
        hidden_size=2304,
        num_heads=16,
        **kwargs,
    )


# 1.51B
def AnyRes_L(cap_dim, canvas_size, codebooks, **kwargs):
    return AnyResolutionModel(
        cap_dim,
        canvas_size,
        codebooks[1:],
        depth=24,
        hidden_size=1152,
        num_heads=16,
        **kwargs,
    )


# 480M
def AnyRes_B(cap_dim, canvas_size, codebooks, **kwargs):
    return AnyResolutionModel(
        cap_dim,
        canvas_size,
        codebooks[1:],
        depth=12,
        hidden_size=1024,
        num_heads=12,
        **kwargs,
    )


# 152.1455M
def AnyRes_S(cap_dim, canvas_size, codebooks, **kwargs):
    return AnyResolutionModel(
        cap_dim,
        canvas_size,
        codebooks[1:],
        depth=12,
        hidden_size=768,
        num_heads=12,
        **kwargs,
    )


# AnyRes_models = {
#     'AnyRes-XL/2': AnyRes_XL_2,  'AnyRes-XL/4': AnyRes_XL_4,  'AnyRes-XL/8': AnyRes_XL_8,
#     'AnyRes-L/2':  AnyRes_L_2,   'AnyRes-L/4':  AnyRes_L_4,   'AnyRes-L/8':  AnyRes_L_8,  "AnyRes-L/16": AnyRes_L_16, "AnyRes-L/32": AnyRes_L_32,
#     'AnyRes-B/2':  AnyRes_B_2,   'AnyRes-B/4':  AnyRes_B_4,   'AnyRes-B/8':  AnyRes_B_8,
#     'AnyRes-S/2':  AnyRes_S_2,   'AnyRes-S/4':  AnyRes_S_4,   'AnyRes-S/8':  AnyRes_S_8,
# }
