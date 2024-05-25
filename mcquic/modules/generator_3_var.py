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
class GeneratorVAR(nn.Module):
    def __init__(
        self,
        channel: int,
        k: int,
        size: List[int],
        denseNorm: bool,
        loadFrom: str,
        **var_args
    ):
        super().__init__()

        self.compressor = Neon(channel, k, size, denseNorm)

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
        from mcquic.data.imagenet_classes import IMAGENET2012_LABELS

        depth = 24

        self.next_residual_predictor: VAR = checkpoint_wrapper(VAR(
            (8, 4096),
            len(IMAGENET2012_LABELS),
            depth=depth,
            embed_dim=1536,
            num_heads=16,
            norm_eps=1e-6,
            attn_l2_norm=True,
            patch_nums=size[::-1],
            drop_path_rate=0.1 * depth/24
        ))
        logging.debug("Created any-res transformer.")

        self.next_residual_predictor.init_weights(init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1)



        # from mcquic.nn import ResidualBlock, ResidualBlockShuffle, ResidualBlockWithStride
        # from mcquic.nn.blocks import AttentionBlock
        # from mcquic.nn.convs import conv3x3, conv1x1, pixelShuffle3x3
        # from mcquic.modules.quantizer import _multiCodebookDeQuantization


        # decoders = list()
        # dequantizers = list()

        # codebook = nn.Parameter(nn.init.trunc_normal_(torch.empty(1, k, self.next_residual_predictor.Cvae), std=math.sqrt(2 / (6 * self.next_residual_predictor.Cvae))))

        # lastSize = size[0] * 2
        # # reverse adding encoder, decoder and quantizer
        # for i, thisSize in enumerate(size):
        #     if thisSize == lastSize // 2:
        #         # codebook = nn.Parameter(nn.init.zeros_(torch.empty(mi, ki, channel // mi)))
        #         # NOTE: quantizer is from large to small, but _freqEMA is from small to large
        #         dequantizer = _multiCodebookDeQuantization(codebook)

        #         restoreHead = pixelShuffle3x3(self.next_residual_predictor.Cvae, self.next_residual_predictor.Cvae, 2)
        #     elif thisSize == lastSize:
        #         dequantizer = _multiCodebookDeQuantization(codebook)

        #         restoreHead = conv3x3(self.next_residual_predictor.Cvae, self.next_residual_predictor.Cvae)
        #     else:
        #         raise ValueError('The given size sequence does not half or equal to from left to right.')


        #     lastSize = thisSize

        #     decoders.append(restoreHead)
        #     dequantizers.append(dequantizer)


        # self._decoders: nn.ModuleList = nn.ModuleList(decoders)

        # self._dequantizers: nn.ModuleList = nn.ModuleList(dequantizers)

        self.input_transform = nn.Identity() # nn.LayerNorm(self.next_residual_predictor.Cvae, 1e-6)

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


        self.compressor.eval()


    def residual_forward(self, code: torch.Tensor, formerLevel: torch.Tensor, level: int):
        if formerLevel is None and level > 0:
            raise RuntimeError('For reconstruction after level-0, you should provide not None formerLevel as input.')
        if formerLevel is not None and level == 0:
            raise RuntimeError('For reconstruction at level-0, you should provide None formerLevel as input.')
        decoder, dequantizer = self._decoders[-(level+1)], self._dequantizers[-(level+1)]
        quantized = dequantizer.decode(code)
        return decoder(quantized + formerLevel) if formerLevel is not None else decoder(quantized)


    def train(self, mode: bool = True):
        retValue = super().train(mode)
        self.compressor.eval()
        # self.text_encoder.eval()
        return retValue

    def forward(self, image, condition: torch.Tensor):
        if self.training:
            ###################### Preparing inputs #########################
                # list of [n, 1, h, w], len of list == levels
                # from low resolution to high resolution
                # NOTE: for reflection padding, the input tensor size (`.numel()`) should not exceed 2^32
                # NOTE: therefore, we manually split image into batch 16
            with torch.autocast("cuda", enabled=False):
                with torch.no_grad():
                    codes = self.compressor.encode(image.float())
                    all_forwards_for_residual = list()
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

            new_all_forwards_for_residual = list()
            for x in all_forwards_for_residual:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n, h*w, -1)
                new_all_forwards_for_residual.append(x.to(torch.bfloat16))


            new_all_forwards_for_residual = torch.cat(new_all_forwards_for_residual, 1)
            new_all_forwards_for_residual.requires_grad_()

            rawPredictions = self.next_residual_predictor(
                # [B], [B, L, D]
                condition, self.input_transform(new_all_forwards_for_residual)
            )

            loss = list()
            curIdx = 0
            predictions = list()
            for gt in codes:
                bs, h, w = gt.shape
                pre = rawPredictions[:, curIdx : curIdx + (h * w)]
                pre = pre.permute(0, 2, 1).reshape(bs, -1, h, w)
                predictions.append(pre)
                loss.append((h * w, F.cross_entropy(pre, gt, reduction="none", label_smoothing=0.0)))
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
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                restored = self.compressor.decode(restoredCodes)

            # first_level: [n, k, h, w]
            # predictions: list of [n, k, h, w], len of list == levels - 1 (give previous embedding, predict next code)
            return (
                [*predictions],
                sum([(l).sum() for hw, l in loss]) / len(image),
                codes,
                restored,
                [l.mean() for _, l in loss],
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
                class_embed = self.class_pos_embed[condition]
                # given shape and condition, produce token with secified shape
                h, w = 1, 1 # first scale is 1x1
                bs, hidden_size = class_embed.shape
                # ================= start loop =================
                first_level_token = self.next_residual_predictor((None, 0), class_embed)
                first_level_token = first_level_token.unsqueeze(dim=1)
                first_level_token = first_level_token.permute(0, 2, 1).reshape(
                    bs, -1, h, w
                )

                first_scale_feat = self.compressor.residual_forward(
                    first_level_token, None, 0
                )

                predictions = [first_level_token]
                input_feats = [first_scale_feat]
                former_level_feat = first_scale_feat.clone()

                for i in range(1, len(self.compressor.Codebooks)):
                    # [bs, h * w]
                    next_level_token = self.next_residual_predictor(
                        (input_feats, i), class_embed
                    )
                    # get current scale
                    scale = int(math.sqrt(next_level_token.shape[-1]))
                    h, w = scale, scale
                    # [bs, 1, h, w]
                    next_level_token = next_level_token.reshape(bs, h, w).unsqueeze(1)

                    # [bs, tok_dim, h, w]
                    next_scale_feat = self.compressor.residual_forward(
                        next_level_token, former_level_feat, i
                    )
                    former_level_feat = next_scale_feat.clone()
                    predictions.append(next_level_token)
                    input_feats.append(next_scale_feat)

                # # list of [bs, hi, wi]
                # predictions.insert(0, first_level)
                # # list of [bs, 1, hi, wi]
                # predictions = [p.unsqueeze(1) for p in predictions]
                restored = self.compressor.decode(predictions)

                return predictions, restored



import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn as nn
from torch.nn import functional as F


def sample_with_top_k_top_p_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz torch.multinomial can only be used for 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


def gumbel_softmax_with_rng(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1, rng: torch.Generator = None) -> torch.Tensor:
    if rng is None:
        return F.gumbel_softmax(logits=logits, tau=tau, hard=hard, eps=eps, dim=dim)

    gumbels = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_(generator=rng).log())
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):    # taken from timm
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'(drop_prob=...)'


# this file only provides the 3 blocks used in VAR transformer
__all__ = ['FFN', 'AdaLNSelfAttn', 'AdaLNBeforeHead']


# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
# try:
#     from flash_attn.ops.layer_norm import dropout_add_layer_norm
#     from flash_attn.ops.fused_dense import fused_mlp_func
# except ImportError: pass
# # automatically import faster attention implementations
# try: from xformers.ops import memory_efficient_attention
# except ImportError: pass
# try: from flash_attn import flash_attn_func              # qkv: BLHc, ret: BLHcq
# except ImportError: pass
# try: from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
# except ImportError:
#     def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
#         attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
#         if attn_mask is not None: attn.add_(attn_mask)
#         return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value

from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation='gelu_approx', save_pre_act=self.training, return_residual=False, checkpoint_lvl=0,
                heuristic=0, process_group=None,
            ))
        else:
            return self.drop(self.fc2( self.act(self.fc1(x)) ))

    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'


class SelfAttention(nn.Module):
    def __init__(
        self, block_idx, embed_dim=768, num_heads=12,
        attn_drop=0., proj_drop=0., attn_l2_norm=False, flash_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads  # =64
        self.attn_l2_norm = attn_l2_norm
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)

        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: bool): self.caching, self.cached_k, self.cached_v = enable, None, None

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, attn_bias):
        B, L, C = x.shape

        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype
        # qkv: BL3Hc

        using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        if using_flash or self.using_xform: q, k, v = qkv.unbind(dim=2); dim_cat = 1   # q or k or v: BLHc
        else: q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); dim_cat = 2               # q or k or v: BHLc

        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform: scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        if self.caching:
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat); v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)

        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), attn_bias=None if attn_bias is None else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1), p=dropout_p, scale=self.scale).view(B, L, C)
        else:
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)

        return self.proj_drop(self.proj(oup))
        # attn = (q @ k.transpose(-2, -1)).add_(attn_bias + self.local_rpb())  # BHLc @ BHcL => BHLL
        # attn = self.attn_drop(attn.softmax(dim=-1))
        # oup = (attn @ v).transpose_(1, 2).reshape(B, L, -1)     # BHLL @ BHLc = BHLc => BLHc => BLC

    def extra_repr(self) -> str:
        return f'using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}'


class AdaLNSelfAttn(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
        flash_if_available=False, fused_if_available=True,
    ):
        super(AdaLNSelfAttn, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = SelfAttention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available)
        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=drop, fused_if_available=fused_if_available)

        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)

        self.fused_add_norm_fn = None

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, cond_BD, attn_bias):   # C: embed_dim, D: cond_dim
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        x = x + self.drop_path(self.attn( self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1), attn_bias=attn_bias ).mul_(gamma1))
        x = x + self.drop_path(self.ffn( self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed when FusedMLP is used
        return x

    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}'


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2*C))

    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)



import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, codebook_size,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = codebook_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads

        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training

        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=torch.cuda.current_device())

        # 1. input (word) embedding
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=torch.cuda.current_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng

        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))

        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn*pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        for b in self.blocks: b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]

    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)

            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)

        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated

        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


# class VARHF(VAR, PyTorchModelHubMixin):
#             # repo_url="https://github.com/FoundationVision/VAR",
#             # tags=["image-generation"]):
#     def __init__(
#         self,
#         vae_kwargs,
#         num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
#         norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
#         attn_l2_norm=False,
#         patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
#         flash_if_available=True, fused_if_available=True,
#     ):
#         vae_local = VQVAE(**vae_kwargs)
#         super().__init__(
#             vae_local=vae_local,
#             num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
#             norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
#             attn_l2_norm=attn_l2_norm,
#             patch_nums=patch_nums,
#             flash_if_available=flash_if_available, fused_if_available=fused_if_available,
#         )
