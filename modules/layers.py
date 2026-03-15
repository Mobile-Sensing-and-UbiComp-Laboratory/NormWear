# Copyright (c) School of Computing, Information, and Data Science, University of California San Diego.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import math
import numpy as np
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

from itertools import repeat
import collections.abc

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

class CheckShape(nn.Module):
    def __init__(self, remark, key=None):
        super().__init__()
        self.remark = remark
        self.key = key
    def forward(self, x, **kwargs):
        if self.remark is not None:
            print(self.remark, x.shape)
        
        out = x
        if self.key is not None:
            out = self.key(x)
        return out

# fix time position embedding
class tAPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048, scale_factor=1.0, trainable=False):
        super(tAPE, self).__init__()
        self.max_len = max_len
        self.trainable = trainable
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

        # trainable parameter
        if self.trainable:
            self.trainable_pe = nn.Parameter(torch.zeros(pe.shape))
    
    def interpolate_pe(self, original_pe, target_len):
        # original_pe: (1, original_length, embedding_size)
        # return interpolated_pe: (1, target_len, embedding_size)
        # fetch required info
        original_len = original_pe.size(1)
        if target_len <= original_len: # if shorted then just clip
            return original_pe.unfold(dimension=1, size=target_len, step=1).mean(dim=1).permute(0, 2, 1)
            # return original_pe[:, :target_len, :]
        
        # interpolate
        pe_reshaped = original_pe.permute(0, 2, 1) # 1, embedding_size, original_length
        pe_interpolated = F.interpolate(
            pe_reshaped,
            size=target_len,  # target length
            mode='nearest-exact',
            # align_corners=True  # casual scenario is recommended to be true
        )
        interpolated_pe = pe_interpolated.permute(0, 2, 1) # 1, original_length, embedding_size
        return interpolated_pe

    def cyclic_pe(self, original_pe, target_len):
        # original_pe: (1, original_length, embedding_size)
        # return interpolated_pe: (1, target_len, embedding_size)
        
        # cycling
        # pe_reshaped = original_pe.permute(0, 2, 1) # 1, embedding_size, original_length
        cyclic_pe = torch.concat((original_pe, original_pe), dim=1) # 1, original_length*2, embedding_size
        while cyclic_pe.shape[-1] < target_len:
            cyclic_pe = torch.concat((cyclic_pe, original_pe), dim=1)
        # cyclic_pe = pe_reshaped.permute(0, 2, 1) # 1, original_length, embedding_size

        # clip
        if target_len <= cyclic_pe.shape[1]: # if shorted then just clip
            return cyclic_pe[:, :target_len, :]
        return cyclic_pe

    def duplicate_pretrained_pe(self, pretrained_end_idx=256-16):
        # self.pe shape: [1, max_length, embedding_size]
        # self.trainable_pe shape: [1, max_length, embedding_size]
        # NOTE: This function will be called after pretrained pe get loaded
        # TODO: The index from 0 to pretrained_end_idx are well-pretrained, and the rest remain randomly initialized. 
        # when this function get called, duplicate the parameters values from 0 to pretrained_end_idx to all the later indeces, do for both pe and trainable pe
        with torch.no_grad():
            for param in [self.pe, self.trainable_pe]:
                # param shape: [1, max_length, embedding_size]
                max_len = param.shape[1]

                pretrained = param[:, :pretrained_end_idx, :].clone()

                remaining = max_len - pretrained_end_idx
                if remaining <= 0:
                    continue

                # repeat pretrained block enough times
                repeat_factor = int(((remaining + pretrained_end_idx - 1) / pretrained_end_idx)+1)
                tiled = pretrained.repeat(1, repeat_factor, 1) # 1, repeat_factor*pretrained_len, embedding_size

                # fill the remaining positions
                param[:, pretrained_end_idx:, :] = tiled[:, :remaining, :]


    def forward(self, x): # N, L, C
        has_four_dim = False
        if len(x.shape) == 4:
            has_four_dim = True
            bn, nvar, L, C = x.shape
            x = x.reshape(bn*nvar, L, C)
        
        # adjust pe function
        pe_adjust = self.interpolate_pe # seems work better than cyclic
        # pe_adjust = self.cyclic_pe

        # NOTE: this is just because the very 1st version has false length, remove this afterward
        curr_max_len = self.max_len if self.max_len < 1024 else 256-16

        # add position embeddings
        x = x + pe_adjust(self.pe[:, :curr_max_len, :], x.shape[1])
        # x = x + pe_adjust(self.pe[:, :, :], x.shape[1])
        # x = x + self.pe[:, pe_start_idx:pe_start_idx+x.shape[1], :]
        if self.trainable:
            x = x + pe_adjust(self.trainable_pe[:, :curr_max_len, :], x.shape[1])
            # x = x + self.trainable_pe[:, pe_start_idx:pe_start_idx+x.shape[1], :]
        x = self.dropout(x)

        if has_four_dim:
            x = x.reshape(bn, nvar, L, C)
        return x

# fix position embedding
def get_2d_sincos_pos_embed_flexible(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
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
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def interpolate_pos_embed(model, checkpoint_model,orig_size=(43,13),new_size=(43,13)): 
    '''
    Input: model: the class is definging for downstream
           checkpoint_model: pre-train weight
           orig_size = (old_num_time_patches,old_num_freq_patches) = (43,13)
    '''

    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed'] # 1 x 560 x 768 (1 x num_patches x E)
        embedding_size = pos_embed_checkpoint.shape[-1] # 768

        # number of special tokens (e.g. in this case num_extra_tokens = 1 for the cls token)
        num_patches = model.patch_embed.num_patches  
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches 
        
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size[0], orig_size[1], new_size[0], new_size[1]))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:] # old positions
            pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size[0], new_size[1]), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 embed_dim=768, 
                 stride=16,
                 dropout=0.1, # position_embed params
                 scale_factor=1.0,
                 use_tAPE=False):
        super().__init__()

        '''
        For pretrain:
        we fixed img_size to be (387,65)
        thereby using patch_size (9,5)
        yielding 43 * 13 = 559 patches
        '''

        '''
        for downstream task,
        resize 
        '''

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.use_tAPE = use_tAPE
        self.img_size = img_size
        self.patch_size = patch_size
        

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) 
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        
        self.num_patches = h*w
        if use_tAPE:
            self.time_pos_embed = tAPE(d_model=embed_dim, max_len=h, 
                                    dropout=dropout,scale_factor=scale_factor)

    def get_output_shape(self, img_size, device=torch.device('cpu')):
        # todo: don't be lazy..
        return self.proj(torch.randn(1,3,img_size[0],img_size[1]).to(device)).shape 

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x) # 32, 3, 387, 65 -> 32, 768, 43, 13
        if self.use_tAPE:
            x = self.time_pos_embed(x) # 32, 768, 43, 13
        x = x.flatten(2) # 32, 768, 43, 13 -> 32, 768, 559
        x = x.transpose(1, 2) # 32, 768, 215 -> 32, 559, 768
        return x
    



class PatchEmbed_ts(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, ts_len=387, 
                 patch_size=9, 
                 embed_dim=768, 
                 stride=9,
                 dropout=0.1, # position_embed params
                 scale_factor=1.0,):
        super().__init__()

        '''
        For pretrain:
        we fix length and nvar -> bs*nvar x L = bs *4 x 388
        '''
        
        self.ts_len = ts_len
        self.patch_size = patch_size
        

        self.proj = conv1d_layer = nn.Conv1d(in_channels=1,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size)


        bs, E, P = self.get_output_shape(ts_len) # n, emb_dim, P

        self.patch_hw = patch_size
        self.num_patches = P
        
    def get_output_shape(self, ts_len):
        # todo: don't be lazy..
        return self.proj(torch.randn(1,1,ts_len)).shape # bs, num_parches, L

    def forward(self, x):
        bs, L = x.shape
        x = x.unsqueeze(1)  # bs, 1, L

        x = self.proj(x) # bs,E,L
        x = x.permute(0, 2, 1) # bs, L, E

        return x



class VAE_Latent(nn.Module):
    def __init__(self, emb_size, out_size, bias=None):
        super().__init__()

        self.mu = nn.Linear(emb_size, out_size, bias=bias)
        self.var = nn.Sequential(
            nn.Linear(emb_size, out_size, bias=bias),
            nn.Softplus()
        )
        
    def forward(self, x):
        if not self.training:
            # during inference, just return the mean
            return self.mu(x)
        
        # generate mean and variance
        mu, var = self.mu(x), self.var(x)

        # reparametrization trick
        eps = torch.randn_like(var)
        z = mu + var*eps
        return z

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
            vae_out=False,
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

        # final out linear
        if not vae_out:
            self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        else:
            self.fc2 = VAE_Latent(hidden_features, out_features, bias=bias[1])


        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SwiGLU_Mlp(nn.Module):
    """
    SwiGLU MLP block used in modern transformers (LLaMA, Qwen).
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        norm_layer=None,
        act_layer=None,
        bias=True,
        drop=0.,
        use_conv=False,
        vae_out=False,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 4)  # typical MLP ratio

        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        # SwiGLU uses TWO projections
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.fc2 = linear_layer(in_features, hidden_features, bias=bias[0])

        self.norm = norm_layer(hidden_features, eps=1e-06) if norm_layer is not None else nn.Identity()

        # final projection
        if not vae_out:
            self.fc3 = linear_layer(hidden_features, out_features, bias=bias[1])
        else:
            self.fc3 = VAE_Latent(hidden_features, out_features, bias=bias[1])

        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):

        gate = F.silu(self.fc1(x))     # SiLU activation
        value = self.fc2(x)

        x = gate * value               # SwiGLU gating

        x = self.norm(x)

        x = self.fc3(x)

        x = self.drop2(x)

        return x

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            use_casual: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = use_fused_attn()
        self.fused_attn = True
        self.use_casual = use_casual

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim, eps=1e-06) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, eps=1e-06) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # reservor adjacency matrix
        self.rc_attn = None

    def forward(
        self, 
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # kv cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)   # [B, h, past+N, d]
            v = torch.cat([past_v, v], dim=2)

        # whether to use scaled attn or raw attn
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                is_causal=self.use_casual
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        # mlp layers
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def scaled_dot_product_attention_kvcache(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            use_casual: bool = False,
            vae_out: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-06)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            use_casual=use_casual,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim, eps=1e-06)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            vae_out=vae_out,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
    
class PatchTSTKernelEmbeddingLocal(nn.Module):
    def __init__(self, poly_degrees=2, num_poly_feats=120, patch_length=16, rff_scale=1.0, num_rff=256, rff_trainable=False, d_feat=512, d_out=512):
        super().__init__()
        poly_degrees_lst = range(2, 2 + poly_degrees)

        self.num_poly_feats = num_poly_feats
        self.patch_indices = [
            torch.randint(
                high=patch_length,
                size=(self.num_poly_feats, d),
                requires_grad=False,
            )
            for d in poly_degrees_lst
        ]
        self.freq_weights = nn.Parameter(
            rff_scale * torch.randn(patch_length, num_rff // 2),
            requires_grad=rff_trainable,
        )
        self.freq_biases = nn.Parameter(
            torch.randn(1, 1, 1, num_rff // 2),
            requires_grad=rff_trainable,
        )
        self.projection = nn.Linear(d_feat, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input for embedding
        return:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, d_model)`
        """

        poly_feats = [x[..., pis].prod(dim=-1) for pis in self.patch_indices]

        weighted_x = x @ self.freq_weights + self.freq_biases
        rff_feats = torch.cat([torch.sin(weighted_x), torch.cos(weighted_x)], dim=-1)

        # features = torch.cat([cdiff_feats, *poly_feats, rff_feats], dim=-1)
        features = torch.cat([x, *poly_feats, rff_feats], dim=-1)
        # print(features.shape)
        # exit()
        features = self.projection(features)
        return features