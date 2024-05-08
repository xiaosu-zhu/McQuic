from typing import List, Union
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
import transformers
import transformers.modeling_outputs

from mcquic.modules.compressor import Neon

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


class Generator(nn.Module):
    def __init__(self, channel: int, k: List[int], denseNorm: bool, loadFrom: str, *_, **__):
        super().__init__()
        self.compressor = Neon(channel, k, denseNorm)
        state_dict = torch.load(loadFrom, map_location='cpu')
        self.compressor.load_state_dict({k[len('module._compressor.'):]: v for k, v in state_dict['trainer']['_model'].items()})
        for params in self.compressor.parameters():
            params.requires_grad_(False)
        logging.info('Loaded compressor checkpoint from %s.', loadFrom)

        logging.debug('Start loading clip...')
        self.text_encoder = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=False)
        logging.debug('Loaded clip text model from %s.', "openai/clip-vit-base-patch32")
        self.text_tokenizer = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=False)
        logging.debug('Loaded clip text model from %s.', "openai/clip-vit-base-patch32")
        for params in self.text_encoder.parameters():
            params.requires_grad_(False)

        clip_text_channels = self.text_encoder.text_model.config.hidden_size

        # NOTE: text_to_first_level: This transforms text embeddings to the first level token
        # NOTE: next_residual_predictor: we only need first (level - 1) codebook, and corresponding canvas.
        # NOTE: remove first dim of codebook, since it is for product quantization
        self.text_to_first_level, self.next_residual_predictor = AnyRes_S(clip_text_channels, [4, 8, 16, 32], [codebook.squeeze(0) for codebook in self.compressor.Codebooks])
        logging.debug('Created any-res transformer.')


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
                all_backwards_for_residual = list()
                for sp in splitted:
                    codes = self.compressor.encode(sp)
                    allCodes.append(codes)
                    # list of [n, c, 2h, 2w]
                    all_backwards_for_residual.append([self.compressor.residual_backward(code, level) for level, code in enumerate(codes[:-1])])
                codes = [torch.cat(x) for x in zip(*allCodes)]
                # list - 1 of [n, c, 2h, 2w]
                all_backwards_for_residual = [torch.cat(x) for x in zip(*all_backwards_for_residual)]

                # input_ids: [B, max_len] int ids, where `49407` for padding
                # attention_mask: [B, max_len] {0, 1}. where `1` for valid, `0` for padding mask
                batch_encoding = self.text_tokenizer(text=condition, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')

                input_ids = batch_encoding.input_ids.to(image.device)
                attention_mask = batch_encoding.attention_mask.to(image.device)

                # last_hidden_state: [B, max_len, D]
                # pooler_output: [B, D]
                text_embedding: transformers.modeling_outputs.BaseModelOutputWithPooling = self.text_encoder(input_ids, attention_mask=attention_mask, return_dict=True)

            # NOTE: remove product quantization artifacts, since we don't use product quantization
            codes = [c.squeeze(1) for c in codes]

            # given shape and condition, produce token with secified shape
            first_level = self.text_to_first_level(codes[0].shape, text_embedding.pooler_output.detach().clone(), text_embedding.last_hidden_state.detach().clone(), attention_mask)

            predictions = self.next_residual_predictor(all_backwards_for_residual, text_embedding.pooler_output.detach().clone(), text_embedding.last_hidden_state.detach().clone(), attention_mask)

            loss = [F.cross_entropy(pre, gt, reduction='none') for pre, gt in zip([first_level, *predictions], codes)]
            # loss = [F.cross_entropy(pre, gt, reduction='none') for (pre, gt) in zip([first_level, *predictions], codes)]

            # list of [n, 1, h, w], len of list == levels
            restoredCodes = [pre.detach().clone().argmax(1, keepdim=True) for pre in predictions]
            # [n, 1, h, w]
            restoredCodes.insert(0, first_level.detach().clone().argmax(1, keepdim=True))
            with torch.no_grad():
                splitted = list(zip(*list(torch.split(x, 16) for x in restoredCodes)))

                allRestored = list()
                for sp in splitted:
                    allRestored.append(self.compressor.decode(sp))
                restored = torch.cat(allRestored)

            # first_level: [n, k, h, w]
            # predictions: list of [n, k, h, w], len of list == levels - 1 (give previous embedding, predict next code)
            return [first_level, *predictions], sum([l.sum() / len(image) for l in loss]), codes, restored, [l.mean() for l in loss]
        else:
            ###################### Preparing inputs #########################
            with torch.no_grad():
                device = next(self.parameters()).device
                # input_ids: [B, max_len] int ids, where `49407` for padding
                # attention_mask: [B, max_len] {0, 1}. where `1` for valid, `0` for padding mask
                batch_encoding = self.text_tokenizer(text=condition, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')

                input_ids = batch_encoding.input_ids.to(device)
                attention_mask = batch_encoding.attention_mask.to(device)

                # last_hidden_state: [B, max_len, D]
                # pooler_output: [B, D]
                text_embedding: transformers.modeling_outputs.BaseModelOutputWithPooling = self.text_encoder(input_ids, attention_mask=attention_mask, return_dict=True)

                # given shape and condition, produce token with secified shape
                first_level = self.text_to_first_level([len(text_embedding), 2, 2], text_embedding.pooler_output.detach().clone(), text_embedding.last_hidden_state.detach().clone(), attention_mask)

                backward_for_reidual = self.compressor.residual_backward(first_level.unsqueeze(1), 0)


                predictions = list()
                predictions.append(self.next_residual_predictor((backward_for_reidual, 0), text_embedding.pooler_output.detach().clone(), text_embedding.last_hidden_state.detach().clone(), attention_mask))

                for i in range(1, len(self.compressor.Codebooks) - 1):
                    backward_for_reidual = self.compressor.residual_backward(predictions[-1].unsqueeze(1), i)
                    predictions.append(self.next_residual_predictor((backward_for_reidual, i), text_embedding.pooler_output.detach().clone(), text_embedding.last_hidden_state.detach().clone(), attention_mask))

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
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
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
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
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

# class SelfAttention(nn.Module):
#     def __init__(
#             self,
#             dim: int,
#             num_heads: int = 8,
#             attn_drop: float = 0.,
#     ) -> None:
#         super().__init__()
#         self.attention = nn.MultiheadAttention(dim, num_heads, attn_drop, batch_first=True)

#         # qkv packed attention
#         # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         # # self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         # # self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         # self.proj = nn.Sequential(
#         #     nn.Linear(dim, dim),
#         #     nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()
#         # )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.attention(x, x, x)
#         # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)#.permute(2, 0, 3, 1, 4)
#         # # q, k, v = qkv.unbind(0)
#         # # q, k = self.q_norm(q), self.k_norm(k)

#         # if USE_FLASH_ATTN:

#         #     with torch.autocast(device_type="cuda", dtype=torch.float16):
#         #         # [B, N, n_head, head_dim]
#         #         out: torch.Tensor = flash_attn_qkvpacked_func(qkv.to(torch.float16), self.attn_drop, self.scale).to(torch.float32).reshape(B, N, C).contiguous()
#         # else:
#         #     # [B, N, nheads, head_dim]
#         #     query, key, value = torch.chunk(qkv, 3, 2)
#         #     F.scaled_dot_product_attention()

#         return out

# class CrossAttention(nn.Module):
#     def __init__(
#             self,
#             dim: int,
#             num_heads: int = 8,
#             attn_drop: float = 0.,
#             proj_drop: float = 0.,
#     ) -> None:
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.attn_drop = attn_drop

#         # qkv packed attention
#         self.wq = nn.Linear(dim, dim, bias=qkv_bias)
#         self.wk = nn.Linear(dim, dim, bias=qkv_bias)
#         self.wv = nn.Linear(dim, dim, bias=qkv_bias)
#         # self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         # self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.proj = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()
#         )

#     # attention_mask: [B, max_len] {1, 0}, `1` for valid, `0` for masked
#     def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask) -> torch.Tensor:
#         # [B, N, nheads, ndims]
#         q = self.wq(q).reshape(q.shape[0], q.shape[1], self.num_heads, -1)
#         k = self.wk(k).reshape(k.shape[0], k.shape[1], self.num_heads, -1)
#         v = self.wv(v).reshape(v.shape[0], v.shape[1], self.num_heads, -1)

#         ##### Concat query ####
#         B, N, nhead, ndim = q.shape
#         q = q.reshape(B*N, nhead, ndim)
#         # [B+1], cumulative sequence length
#         # start from 0, end to [B * q_len]
#         cum_q_len = torch.arange(0, (B+1)*N, step=N, dtype=torch.int32, device=q.device)

#         ##### Concat valid key and value ####
#         indexing = attention_mask.bool()
#         # [B, max_len] to [sum(seq_len)]
#         k = k[indexing]
#         v = v[indexing]
#         # [B+1], cumulative sequence length
#         # start from 0, end to [sum(k_len)]
#         cum_seq_len = attention_mask.sum(-1).cumsum(0, dtype=torch.int32)
#         # insert a 0 at start
#         cum_k_len = torch.cat([torch.zeros([1], dtype=cum_seq_len.dtype, device=cum_seq_len.device), cum_seq_len])

#         with torch.autocast(device_type="cuda", dtype=torch.float16):
#             # [B*N, n_head, head_dim]
#             out: torch.Tensor = flash_attn_varlen_func(q.to(torch.float16), k.to(torch.float16) ,v.to(torch.float16), cum_q_len, cum_k_len, N, attention_mask.sum(-1).max(), self.attn_drop, self.scale).to(torch.float32)

#         out = out.reshape(B, N, nhead, ndim).reshape(B, N, -1)

#         return self.proj(out)


class TransformerBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Zero-out adaLN modulation layers in DiT blocks:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, condition):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(6, dim=1)
        attn_qkv = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(attn_qkv, attn_qkv, attn_qkv, need_weights=False)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class CrossTransformerBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0., proj_drop=0.):
        super().__init__()
        # self.normq = nn.LayerNorm(hidden_size, eps=1e-6)
        # self.normk = nn.LayerNorm(hidden_size, eps=1e-6)
        # self.normv = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=proj_drop)
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        # )

    # attention_mask: [B, max_len] {1, 0}, `1` for valid, `0` for masked
    def forward(self, x, condition, attention_mask):
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + self.attn(x, condition, condition, key_padding_mask=(attention_mask!=1), need_weights=False)[0]

        x = x + self.mlp(self.norm2(x))
        return x


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
    def __init__(self, hidden_size, prediction_num, prototypes = None):
        super().__init__()
        if prototypes is not None:
            self.register_buffer('prototypes', prototypes)
            self.prototype_proj = nn.Linear(prototypes.shape[-1], hidden_size)
            self.norm_final = nn.InstanceNorm2d(hidden_size, affine=False, eps=1e-6)
            self.linear = nn.Conv2d(hidden_size, hidden_size, 1)
            self.skip_connection = nn.Conv2d(hidden_size, hidden_size, 1)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size)
            )

        else:
            self.norm_final = nn.InstanceNorm2d(hidden_size, affine=False, eps=1e-6)
            self.linear = nn.Conv2d(hidden_size, prediction_num, 1)
            self.skip_connection = nn.Conv2d(hidden_size, prediction_num, 1)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size)
            )

        # Zero-out adaLN modulation layers in DiT blocks:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)


    def forward(self, x, condition):
        if hasattr(self, 'prototypes'):
            # [B, D]
            shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
            # modulate, [B, D, H, W]
            out = self.norm_final(x) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            # [B, D, H, W]
            x = self.linear(out) + self.skip_connection(x)
            n, c, h, w = x.shape
            # [K, D]
            codes = self.prototype_proj(self.prototypes)
            # [b, hw, d] @ [b, d, k] -> [b, hw, k]
            similarity = torch.bmm(x.permute(0, 2, 3, 1).reshape(n, h*w, c).contiguous(), codes.expand(n, *codes.shape).permute(0, 2, 1).contiguous())
            # [b, k, h, w]
            similarity = similarity.reshape(n, h, w, -1).permute(0, 3, 1, 2).contiguous()
            return similarity
        else:
            # [B, D]
            shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
            # modulate, [B, D, H, W]
            out = self.norm_final(x) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            x = self.linear(out) + self.skip_connection(x)
            return x


class AnyResolutionBlock(nn.Module):
    """
    Next scale model with a Transformer backbone.
    """
    def __init__(
        self,
        codebook,
        canvas_size, # e.g.: 32, corresponding to raw pixels: 512
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        # learn_sigma=True,
    ):
        super().__init__()
        input_dim = codebook.shape[-1]
        # self.learn_sigma = learn_sigma
        self.in_channels = input_dim
        self.canvas_size = canvas_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # self.input_embedding = nn.Parameter(codebook.detach().clone())
        self.input_transform = nn.Conv2d(input_dim, hidden_size, 1, bias=False)
        self.pre_layer = checkpoint_wrapper(
            FinalLayer(hidden_size, hidden_size)
        )
        self.final_layer = checkpoint_wrapper(
            FinalLayer(hidden_size, len(codebook), codebook)
        )


        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.num_patches = canvas_size * canvas_size * 64 # expand canvas to 8 * canvas, to support at least 8*resolution
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            checkpoint_wrapper(TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)) for _ in range(depth)
        ])
        self.condition_blocks = nn.ModuleList([
            checkpoint_wrapper(CrossTransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)) for _ in range(depth)
        ])
        self.proj_layer = checkpoint_wrapper(ProjLayer(hidden_size, scale_factor=1))
        # self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize transformer layers and proj layer:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Zero-out output layers:
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0)

    def random_pos_embed(self, h, w):
        H = W = int(math.sqrt(self.num_patches))
        # [H, W, D]
        pos_embed = self.pos_embed.reshape(H, W, -1)
        random_up = random.randint(0, H - h - 1)
        random_left = random.randint(0, W - w - 1)
        return pos_embed[random_up:random_up+h, random_left:random_left+w].reshape(h*w, -1)

    def center_pos_embed(self, h, w):
        H = W = int(math.sqrt(self.num_patches))
        # [H, W, D]
        pos_embed = self.pos_embed.reshape(H, W, -1)
        up_start = (H - h) // 2
        left_start = (W - w) // 2
        return pos_embed[up_start:up_start+h, left_start:left_start+w].reshape(h*w, -1)

    def unpatchify(self, x, h, w):
        """
        x: (bs, patch_size**2, 4 * D)
        imgs: (bs, H, W, D)
        """
        bs, hw, dim = x.shape

        return x.permute(0, 2, 1).reshape(bs, dim, h, w).contiguous() # [bs, 4 * D, h, w]
        return self.pixel_shuffle(x) # [bs, D, 2 * h, 2 * w]

    def forward(self, input_embedding, pooled_condition, sequence_condition, attention_mask):
        # 0, 1, 2, 3
        bs, c, h, w = input_embedding.shape
        # [b, h, w, d]
        # x = self.input_embedding[code]
        # [b, h*w, hidden]
        x = self.input_transform(input_embedding)
        x = (self.pre_layer(x, pooled_condition)).permute(0, 2, 3, 1).reshape(bs, h*w, -1).contiguous()

        if self.training:
            # TODO: change to random
            # selected_pos_embed = self.random_pos_embed(h, w)
            selected_pos_embed = self.center_pos_embed(h, w)
        else:
            selected_pos_embed = self.center_pos_embed(h, w)

        x = x + selected_pos_embed # [bs, hw, hidden]
        for block, cross in zip(self.blocks, self.condition_blocks):
            x = block(x, pooled_condition) + cross(x, sequence_condition, attention_mask)
            # print(x.mean(), x.std())
        x = self.proj_layer(x) # [bs, hw, hidden]
        x = self.unpatchify(x, h ,w) # [bs, hidden, 2h, 2w]
        prediction = self.final_layer(x, pooled_condition) # [bs, k, 2h, 2w]
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


class AnyResolutionTransformer(nn.Module):
    def __init__(self,
        # the token should be firstly aligned to text_dimension, then lift to hidden_size
        text_dimension,
        # NOTE: from low resolution to high resolution. this list should be ascending.
        canvas_size: List[int], # e.g.: 32, corresponding to raw pixels: 512
        codebooks,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0):
        super().__init__()
        self.text_dimension = text_dimension
        self.hidden_size = hidden_size
        # NOTE: we only need first (level - 1) blocks
        self.blocks = nn.ModuleList(
            [AnyResolutionBlock(codebook, can, hidden_size, depth, num_heads, mlp_ratio) for (can, codebook) in zip(canvas_size, codebooks)]
        )
        self.text_lift = nn.ModuleList(
            [nn.Linear(text_dimension, hidden_size) for _ in codebooks]
        )
        # self.blocks = nn.ModuleList([AnyResolutionBlock(canvas_size[-1], in_channels, hidden_size, depth, num_heads, mlp_ratio) for _ in canvas_size] * len(canvas_size))

    def forward(self, all_backwards_for_residual, pooled_condition, sequence_condition, attention_mask):
        if self.training:
            if not isinstance(all_backwards_for_residual, list):
                raise RuntimeError('The given training input is not a list.')
            results = list()
            # [bs, c, h*2, w*2]
            for current, block, text_lift in zip(all_backwards_for_residual, self.blocks, self.text_lift):
                # [bs, k, h*2, w*2]
                results.append(block(current, text_lift(pooled_condition), text_lift(sequence_condition), attention_mask))
            # NOTE: in training, the len of reuslts is level - 1
            return results
        else:
            # [bs, c, h*2, w*2]
            current, level = all_backwards_for_residual
            results = list()
            block, text_lift = self.blocks[level], self.text_lift[level]
            predict = block(current, text_lift(pooled_condition), text_lift(sequence_condition), attention_mask)
            # NOTE: in training, the len of reuslts is level - 1
            return predict.argmax(1)



class TextConditionedGenerator(nn.Module):
    def __init__(self,
        text_dimension: int,
        prediction_num: int,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0):
        super().__init__()
        # self.learn_sigma = learn_sigma
        self.text_dimension = text_dimension
        self.canvas_size = 16 # 16 -> max resolution 4096
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.text_lift = nn.Linear(text_dimension, hidden_size, bias=False)
        self.pre_layer = checkpoint_wrapper(FinalLayer(hidden_size, hidden_size))
        self.post_layer = checkpoint_wrapper(ProjLayer(hidden_size, 1))

        # we only need level - 1 final layers.
        self.final_layer = checkpoint_wrapper(
            FinalLayer(hidden_size, prediction_num)
        )

        self.num_patches = self.canvas_size * self.canvas_size
        # For first level, we need to pick random trainable embedding for startup
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=True)
        self.pos_embed = nn.Parameter(nn.init.uniform_(torch.empty(1, self.num_patches, hidden_size), a=-math.sqrt(2 / (5 * hidden_size)), b=math.sqrt(2 / (5 * hidden_size))))

        self.blocks = nn.ModuleList([
            checkpoint_wrapper(TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)) for _ in range(depth)
        ])
        self.condition_blocks = nn.ModuleList([
            checkpoint_wrapper(CrossTransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)) for _ in range(depth)
        ])
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize transformer layers and proj layer:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # nn.init.xavier_normal_(self.pos_embed)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        # for block in self.blocks:
            # nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            # nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0)

    def random_pos_embed(self, h, w):
        H = W = int(math.sqrt(self.num_patches))
        # [H, W, D]
        pos_embed = self.pos_embed.reshape(H, W, -1)
        random_up = random.randint(0, H - h - 1)
        random_left = random.randint(0, W - w - 1)
        return pos_embed[random_up:random_up+h, random_left:random_left+w].reshape(h*w, -1)

    def center_pos_embed(self, h, w):
        H = W = int(math.sqrt(self.num_patches))
        # [H, W, D]
        pos_embed = self.pos_embed.reshape(H, W, -1)
        up_start = (H - h) // 2
        left_start = (W - w) // 2
        return pos_embed[up_start:up_start+h, left_start:left_start+w].reshape(h*w, -1)

    def forward(self, target_shape, pooled_condition, sequence_condition, attention_mask):
        # 0, 1, 2
        bs, h, w = target_shape

        pooled_condition = self.text_lift(pooled_condition)
        sequence_condition = self.text_lift(sequence_condition)

        if self.training:
            # TODO: change to random
            # selected_pos_embed = self.random_pos_embed(h, w)
            selected_pos_embed = self.center_pos_embed(h, w)
        else:
            selected_pos_embed = self.center_pos_embed(h, w)

        x = selected_pos_embed.expand(bs, h*w, self.hidden_size) # [bs, hw, hidden]
        x = self.pre_layer(x.permute(0, 2, 1).reshape(bs, self.hidden_size, h, w).contiguous(), pooled_condition).permute(0, 2, 3, 1).reshape(bs, h*w, self.hidden_size).contiguous()
        for block, cross in zip(self.blocks, self.condition_blocks):
            x = block(x, pooled_condition) + cross(x, sequence_condition, attention_mask)
        x = self.post_layer(x)
        x = x.reshape(bs, h, w, -1).permute(0, 3, 1, 2).contiguous()
        # x = x.permute(0, )
        prediction = self.final_layer(x, pooled_condition) # [bs, k, h, w]
        # print(prediction[0, :, 0, 0].min(), prediction[0, :, 0, 0].max(), prediction[0, :, 0, 0].softmax(-1).min(), prediction[0, :, 0, 0].softmax(-1).max())
        return prediction if self.training else prediction.argmax(1)







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
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                 AnyRes Configs                                #
#################################################################################
# 2.20B
def AnyRes_XL(text_dimension, canvas_size, codebooks, **kwargs):
    return TextConditionedGenerator(text_dimension, len(codebooks[0]), hidden_size=1024, depth=28, num_heads=16, **kwargs), AnyResolutionTransformer(text_dimension, canvas_size, codebooks[1:], depth=28, hidden_size=1024, num_heads=16, **kwargs)
# 1.51B
def AnyRes_L(text_dimension, canvas_size, codebooks, **kwargs):
    return TextConditionedGenerator(text_dimension, len(codebooks[0]), hidden_size=768, depth=24, num_heads=16, **kwargs), AnyResolutionTransformer(text_dimension, canvas_size, codebooks[1:], depth=24, hidden_size=768, num_heads=16, **kwargs)
# 480M
def AnyRes_B(text_dimension, canvas_size, codebooks, **kwargs):
    return TextConditionedGenerator(text_dimension, len(codebooks[0]), hidden_size=576, depth=12, num_heads=12, **kwargs), AnyResolutionTransformer(text_dimension, canvas_size, codebooks[1:], depth=12, hidden_size=576, num_heads=12, **kwargs)
# 136M
def AnyRes_S(text_dimension, canvas_size, codebooks, **kwargs):
    return TextConditionedGenerator(text_dimension, len(codebooks[0]), hidden_size=288, depth=12, num_heads=6, **kwargs), AnyResolutionTransformer(text_dimension, canvas_size, codebooks[1:], depth=12, hidden_size=288, num_heads=6, **kwargs)


# AnyRes_models = {
#     'AnyRes-XL/2': AnyRes_XL_2,  'AnyRes-XL/4': AnyRes_XL_4,  'AnyRes-XL/8': AnyRes_XL_8,
#     'AnyRes-L/2':  AnyRes_L_2,   'AnyRes-L/4':  AnyRes_L_4,   'AnyRes-L/8':  AnyRes_L_8,  "AnyRes-L/16": AnyRes_L_16, "AnyRes-L/32": AnyRes_L_32,
#     'AnyRes-B/2':  AnyRes_B_2,   'AnyRes-B/4':  AnyRes_B_4,   'AnyRes-B/8':  AnyRes_B_8,
#     'AnyRes-S/2':  AnyRes_S_2,   'AnyRes-S/4':  AnyRes_S_4,   'AnyRes-S/8':  AnyRes_S_8,
# }
