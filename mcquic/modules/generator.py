import torch
from torch import nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from mcquic.modules.compressor import Neon

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class Generator(nn.Module):
    def __init__(self, ):
        super().__init__()



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


class Var(nn.Module):
    """
    Next scale model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.num_patches = 16384 # 2048/16 * 2048/16 -> 128x128 grids
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

    def unpatchify(self, x):
        """
        x: (bs, patch_size**2, 4 * D)
        imgs: (bs, H, W, D)
        """
        bs, patch_num, dim = x.size()
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.permute(0, 2, 1)
        x = x.view(bs, dim, h, w) # [bs, 4 * D, h, w]
        imgs = self.pixel_shuffle(x) # [bs, D, 2 * h, 2 * w]

        # x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        # x = torch.einsum('nhwpqc->nchpwq', x)
        # imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x):
        # 0, 1, 2, 3
        bs, dim, h, w = x.size()
        x = x.permute(0, 2, 3, 1) # 要先 permute，否则直接 view 会把 dim, h 强行拆成 h * w
        x = x.view(bs, h * w, dim)

        # todo: randomly put the image into a 2048x2048 canvas and rope
        selected_pos_embed = self.pos_embed[:, :h * w, :]
        x = x + selected_pos_embed # [bs, PxP, D]
        for block in self.blocks:
            x = block(x)
        x = self.proj_layer(x) # [bs, PxP, 4D]
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


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
#                                   DiT Configs                                  #
#################################################################################

def Var_XL_2(**kwargs):
    return Var(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def Var_XL_4(**kwargs):
    return Var(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def Var_XL_8(**kwargs):
    return Var(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def Var_L_2(**kwargs):
    return Var(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def Var_L_4(**kwargs):
    return Var(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def Var_L_8(**kwargs):
    return Var(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def Var_L_16(**kwargs):
    return Var(depth=24, hidden_size=1024, patch_size=16, num_heads=16, **kwargs)

def Var_L_32(**kwargs):
    return Var(depth=24, hidden_size=1024, patch_size=32, num_heads=16, **kwargs)

def Var_B_2(**kwargs):
    return Var(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def Var_B_4(**kwargs):
    return Var(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def Var_B_8(**kwargs):
    return Var(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def Var_S_2(**kwargs):
    return Var(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def Var_S_4(**kwargs):
    return Var(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def Var_S_8(**kwargs):
    return Var(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


Var_models = {
    'Var-XL/2': Var_XL_2,  'Var-XL/4': Var_XL_4,  'Var-XL/8': Var_XL_8,
    'Var-L/2':  Var_L_2,   'Var-L/4':  Var_L_4,   'Var-L/8':  Var_L_8,  "Var-L/16": Var_L_16, "Var-L/32": Var_L_32,
    'Var-B/2':  Var_B_2,   'Var-B/4':  Var_B_4,   'Var-B/8':  Var_B_8,
    'Var-S/2':  Var_S_2,   'Var-S/4':  Var_S_4,   'Var-S/8':  Var_S_8,
}
