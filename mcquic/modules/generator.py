from typing import List, Union
import logging

import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
import random
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from mcquic.modules.compressor import Neon

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class Generator(nn.Module):
    def __init__(self, channel: int, m: List[int], k: List[int], loadFrom: str, *_, **__):
        super().__init__()
        self.compressor = Neon(channel, m, k)
        state_dict = torch.load(loadFrom, map_location='cpu')
        self.compressor.load_state_dict({k[len('module._compressor.'):]: v for k, v in state_dict['trainer']['_model'].items()})
        logging.info('Load compressor checkpoint from %s.', loadFrom)

        for params in self.compressor.parameters():
            params.requires_grad_(False)

        # NOTE: we only need first (level - 1) codebook, and corresponding canvas.
        # NOTE: remove first dim of codebook, since it is for product quantization
        self.generator = AnyRes_S([2, 4, 8, 16], [codebook.squeeze(0) for codebook in self.compressor.Codebooks[:-1]])

    def forward(self, image):
        if self.training:
            with torch.no_grad():
                self.compressor.eval()
                # list of [n, 1, h, w], len of list == levels
                # from low resolution to high resolution
                codes = self.compressor.encode(image)
            # NOTE: remove product quantization artifacts, since we don't use product quantization
            codes = [c.squeeze(1) for c in codes]
            # print('****** CODES:', [c.shape for c in codes], '********')
            predictions = self.generator(codes)

            # list of [n, 1, h, w], len of list == levels
            restoredCodes = [pre.detach().clone().argmax(1, keepdim=True) for pre in predictions]
            restoredCodes.insert(0, codes[0].unsqueeze(1))
            with torch.no_grad():
                restored = self.compressor.decode(restoredCodes)
            # list of [n, k, h, w], len of list == levels - 1 (give previous embedding, predict next code)
            return predictions, codes, restored
        else:
            raise NotImplementedError


#################################################################################
#                            Core Transformer Model                             #
#################################################################################




class TransformerBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        # self.cross_attn = CrossAttention(hidden_size, num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        # )

    def forward(self, x):
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + self.attn(self.norm1(x))

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
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
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
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        # )

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class AnyResolutionBlock(nn.Module):
    """
    Next scale model with a Transformer backbone.
    """
    def __init__(
        self,
        codebook: torch.Tensor,
        canvas_size, # e.g.: 32, corresponding to raw pixels: 512
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        # learn_sigma=True,
    ):
        super().__init__()
        # self.learn_sigma = learn_sigma
        self.in_channels = codebook.shape[-1]
        self.canvas_size = canvas_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.input_embedding = nn.Parameter(codebook.detach().clone())
        self.pre_layer = nn.Conv2d(codebook.shape[-1], hidden_size, 1, bias=False)
        # we only need level - 1 final layers.
        self.final_layer = nn.Conv2d(hidden_size, len(codebook), 1, bias=False)


        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.num_patches = canvas_size * canvas_size * 64 # expand canvas to 8 * canvas, to support at least 8*resolution
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.proj_layer = ProjLayer(hidden_size, scale_factor=4)
        # self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
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

    def unpatchify(self, x, h, w):
        """
        x: (bs, patch_size**2, 4 * D)
        imgs: (bs, H, W, D)
        """
        bs, hw, dim = x.shape

        x = x.permute(0, 2, 1).reshape(bs, dim, h, w) # [bs, 4 * D, h, w]
        return self.pixel_shuffle(x) # [bs, D, 2 * h, 2 * w]

    def forward(self, code):
        # 0, 1, 2, 3
        bs, h, w = code.shape
        # [b, h, w, d]
        x = self.input_embedding[code]
        # [b, h*w, hidden]
        x = self.pre_layer(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).reshape(bs, h*w, -1).contiguous()

        if self.training:
            # TODO: change to random
            # selected_pos_embed = self.random_pos_embed(h, w)
            selected_pos_embed = self.center_pos_embed(h, w)
        else:
            selected_pos_embed = self.center_pos_embed(h, w)

        x = x + selected_pos_embed # [bs, hw, hidden]
        for block in self.blocks:
            x = block(x)
        x = self.proj_layer(x) # [bs, hw, 4hidden]
        x = self.unpatchify(x, h ,w) # [bs, hidden, 2h, 2w]
        prediction = self.final_layer(x) # [bs, k, 2h, 2w]
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
        # NOTE: from low resolution to high resolution. this list should be ascending.
        canvas_size: List[int], # e.g.: 32, corresponding to raw pixels: 512
        codebooks,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        # NOTE: we only need first (level - 1) blocks
        self.blocks = nn.ModuleList(
            [AnyResolutionBlock(codebook, can, hidden_size, depth, num_heads, mlp_ratio) for can, codebook in zip(canvas_size, codebooks)]
        )
        # self.blocks = nn.ModuleList([AnyResolutionBlock(canvas_size[-1], in_channels, hidden_size, depth, num_heads, mlp_ratio) for _ in canvas_size] * len(canvas_size))

    def forward(self, codes):
        if self.training:
            if not isinstance(codes, list):
                raise RuntimeError('The given training input is not a list.')
            results = list()
            for current, block in zip(codes, self.blocks):
                results.append(block(current))
            # NOTE: in training, the len of reuslts is level - 1
            return results
        else:
            raise NotImplementedError
            current = x
            results = [x]
            for block in self.blocks:
                results.append(block(current))
                current = results[-1]
            # NOTE: in inference, the len of results is level
            return results

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
def AnyRes_XL(canvas_size, codebooks, **kwargs):
    return AnyResolutionTransformer(canvas_size, codebooks, depth=28, hidden_size=1152, num_heads=16, **kwargs)
# 1.51B
def AnyRes_L(canvas_size, codebooks, **kwargs):
    return AnyResolutionTransformer(canvas_size, codebooks, depth=24, hidden_size=1024, num_heads=16, **kwargs)
# 480M
def AnyRes_B(canvas_size, codebooks, **kwargs):
    return AnyResolutionTransformer(canvas_size, codebooks, depth=12, hidden_size=768, num_heads=12, **kwargs)
# 136M
def AnyRes_S(canvas_size, codebooks, **kwargs):
    return AnyResolutionTransformer(canvas_size, codebooks, depth=12, hidden_size=384, num_heads=6, **kwargs)


# AnyRes_models = {
#     'AnyRes-XL/2': AnyRes_XL_2,  'AnyRes-XL/4': AnyRes_XL_4,  'AnyRes-XL/8': AnyRes_XL_8,
#     'AnyRes-L/2':  AnyRes_L_2,   'AnyRes-L/4':  AnyRes_L_4,   'AnyRes-L/8':  AnyRes_L_8,  "AnyRes-L/16": AnyRes_L_16, "AnyRes-L/32": AnyRes_L_32,
#     'AnyRes-B/2':  AnyRes_B_2,   'AnyRes-B/4':  AnyRes_B_4,   'AnyRes-B/8':  AnyRes_B_8,
#     'AnyRes-S/2':  AnyRes_S_2,   'AnyRes-S/4':  AnyRes_S_4,   'AnyRes-S/8':  AnyRes_S_8,
# }
