# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block

from .pos_embed import get_2d_sincos_pos_embed_flexible,get_1d_sincos_pos_embed_from_grid
from .patch_embed import PatchEmbed_new,PatchEmbed_ts
import torch.nn.functional as F

class Spatial_recon(nn.Module):
    def __init__(self, nvar=4, embed_dim=512, inter_dim=256):
        super(Spatial_recon, self).__init__()
        self.norm = nn.BatchNorm1d(nvar * embed_dim)
        self.conv1 = nn.Conv1d(in_channels=nvar * embed_dim, out_channels=inter_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=inter_dim, out_channels=nvar, kernel_size=1)
        self.gelu = nn.GELU()

        self.nvar = nvar
    def forward(self, x):
        '''Input
        x: bs*nvar x 512 x L 
        '''
        _, E, L = x.shape
        
        x = torch.reshape(x,(-1,self.nvar*E,L)) # bs, nvar*E, L
        x = self.norm(x)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        
        return x  # bs x nvar x L


class EncoderLayer(nn.Module):
    def __init__(self,embed_dim = 768,
                 norm_layer = nn.LayerNorm, 
                 num_heads=12, 
                 nvar = 4,
                 mlp_ratio=4.0, 
                 qkv_bias=True,
                 drop=0.1,
                 fuse_frequency=2,
                 curr_layer = 0,
                 # fusion scheme
                 no_fusion=False,
                 mean_fuse=False,):
        super().__init__()

        self.no_fusion = no_fusion
        self.mean_fuse = mean_fuse
        
        self.curr_layer = curr_layer
        self.fuse_frequency = fuse_frequency
        self.nvar = nvar
    
        #self.self_attn = self_attn_model.transformer.blocks[curr_layer].eval()
        self.variate_encoder = Block(dim=embed_dim,
                                      num_heads=num_heads,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        norm_layer=norm_layer)

        if self.curr_layer%self.fuse_frequency==0:
            self.cls_fusion = Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                # proj_drop=drop # comment out for low version on jetson nano
            )
        
    def forward(self,x):
        '''
        input: x: bs*n_vars x L+1 x E
        '''
        _, N, E = x.shape

        x_out = self.variate_encoder(x) # bs * nvars, L+1, E
                        
        # cls fusion
        if self.curr_layer%self.fuse_frequency==0 and not self.no_fusion:
            x_out = torch.reshape(x_out, (-1,self.nvar, N, E))   # z: [bs x nvars x num_patch x E]
            patch_tokens = x_out[:,:,1:,:]

            # fetch token
            if self.mean_fuse:
                cls = x_out.mean(dim=2)
            else:
                cls = x_out[:,:,0,:] # bs x n_vars x E

            # forward and replace
            cls = self.cls_fusion(cls).unsqueeze(2) # bs x n_vars x 1 x E

            x_out = torch.cat((cls,patch_tokens),dim=2)
            bs, n_vars, N, E = x_out.shape
            x_out = torch.reshape(x_out,(bs*n_vars,N,E)) #bs * nvars, L+1, E

        return x_out 




class NormWear(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(387,65), patch_size=(9,5), in_chans=3, 
                 target_len = 388,nvar=4,
                 embed_dim=768, decoder_embed_dim=512,
                 depth=12, num_heads=12,decoder_depth=2,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 fuse_freq=2,is_pretrain=True,
                 mask_t_prob=0.6, mask_f_prob=0.5,
                 mask_prob=0.8,mask_scheme='random', use_cwt=True,
                 attn_score=False,
                 comb_freq=False):
        super().__init__()
        
        self.attn_score = attn_score
        self.comb_freq = comb_freq
        
        if self.attn_score:
            self.attn_score_lst = []

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        if use_cwt:
            self.patch_embed = PatchEmbed_new(img_size, patch_size, in_chans, embed_dim,stride=patch_size) # non-overlap patches
        else:
            self.patch_embed = PatchEmbed_ts(img_size[0],patch_size[0],embed_dim,stride=patch_size[0])
        
        
        num_patches = self.patch_embed.num_patches

        self.mask_t_prob = mask_t_prob
        self.mask_f_prob = mask_f_prob
        self.mask_scheme = mask_scheme
        self.use_cwt = use_cwt
        self.nvar = nvar # need to be generalized

        if mask_scheme == 'random':
            self.mask_prob = mask_prob

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding,will init below
        self.encoder_blocks = nn.ModuleList([
            EncoderLayer(embed_dim = embed_dim,
                 norm_layer = norm_layer, 
                 num_heads=num_heads, 
                 mlp_ratio=mlp_ratio, 
                 drop=0.1,
                 nvar = nvar,
                 fuse_frequency=fuse_freq,
                 curr_layer = i,
                 # fusion scheme
                 no_fusion=False, # False
                 mean_fuse=True, # False
                )
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        self.encoder_depth = depth
        # --------------------------------------------------------------------------
        if is_pretrain:
            # --------------------------------------------------------------------------
            # MAE decoder specifics
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.target_len = target_len
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

            
            self.decoder_blocks = nn.ModuleList([
                Block(dim=decoder_embed_dim,num_heads=8,
                    mlp_ratio=mlp_ratio,norm_layer=norm_layer)
                for i in range(decoder_depth)])

                
            # decode feature for reconstruction
            self.temporal_recon = nn.Linear(num_patches + 1,target_len)
            self.spatial_recon = Spatial_recon(nvar=nvar,embed_dim=decoder_embed_dim,inter_dim=embed_dim//2)
            self.decoder_norm = norm_layer(decoder_embed_dim)
            
            self.cosim = nn.CosineSimilarity(dim=-1, eps=1e-6)

            self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.use_cwt:
            pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)  
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            decoder_pos_embed = get_2d_sincos_pos_embed_flexible(self.decoder_pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
         
        else: # 1D series embedding
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1],np.arange(self.pos_embed.shape[-2],dtype=np.float32))  
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            decoder_pos_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_pos_embed.shape[-1],np.arange(self.decoder_pos_embed.shape[-2],dtype=np.float32)) 
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))


        # # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """

        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]

        #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = imgs.shape[2] // ph
        w = imgs.shape[3] // pw
        
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, ph, w, pw))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, ph*pw * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        print(x.shape)
        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]

        h = 43
        w = 13

        #assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * ph, w * pw))

        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        """
        2D: CWT_imgs (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        
        # number of patches along each axis
        T=43
        F=13

        #x = x.reshape(N, T, F, D)
        len_keep_t = int(T * (1 - mask_t_prob))
        len_keep_f = int(F * (1 - mask_f_prob))

        # noise for mask in time
        noise_t = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample aling time
        ids_shuffle_t = torch.argsort(noise_t, dim=1) # ascend: small is keep, large is remove
        ids_restore_t = torch.argsort(ids_shuffle_t, dim=1) 
        ids_keep_t = ids_shuffle_t[:,:len_keep_t]
        # noise mask in freq
        noise_f = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        ids_shuffle_f = torch.argsort(noise_f, dim=1) # ascend: small is keep, large is remove
        ids_restore_f = torch.argsort(ids_shuffle_f, dim=1) 
        ids_keep_f = ids_shuffle_f[:,:len_keep_f] #

        # generate the binary mask: 0 is keep, 1 is remove
        # mask in freq
        mask_f = torch.ones(N, F, device=x.device)
        mask_f[:,:len_keep_f] = 0
        mask_f = torch.gather(mask_f, dim=1, index=ids_restore_f).unsqueeze(1).repeat(1,T,1) # N,T,F
        # mask in time
        mask_t = torch.ones(N, T, device=x.device)
        mask_t[:,:len_keep_t] = 0
        mask_t = torch.gather(mask_t, dim=1, index=ids_restore_t).unsqueeze(1).repeat(1,F,1).permute(0,2,1) # N,T,F
        mask = 1-(1-mask_t)*(1-mask_f) # N, T, F

        # get masked x
        id2res=torch.Tensor(list(range(N*T*F))).reshape(N,T,F).to(x.device)
        id2res = id2res + 999*mask # add a large value for masked elements
        id2res2 = torch.argsort(id2res.flatten(start_dim=1))
        ids_keep=id2res2.flatten(start_dim=1)[:,:len_keep_f*len_keep_t]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        ids_restore = torch.argsort(id2res2.flatten(start_dim=1))
        mask = mask.flatten(start_dim=1)

        return x_masked, mask, ids_restore
    
    def pos_adjust(self, new_img_size, device=torch.device('cpu')):
        orig_size = (43, 13)
        # new_size = ((new_L // 9), 13)

        _, _, h, w = self.patch_embed.get_output_shape(new_img_size, device=device)
        new_size = (h, w)

        pos_embed_checkpoint = self.pos_embed # 1 x 560 x 768 (1 x num_patches x E)
        embedding_size = pos_embed_checkpoint.shape[-1] # 768

        # number of special tokens (e.g. in this case num_extra_tokens = 1 for the cls token)
        num_extra_tokens = 1
        
        if orig_size != new_size:
            # print("Position interpolate from %dx%d to %dx%d" % (orig_size[0], orig_size[1], new_size[0], new_size[1]))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:] # old positions
            pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size[0], new_size[1]), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            return new_pos_embed
        return self.pos_embed

    def combine_freq_patch(self, x):
        # x: (N, N_var, L, E)
        # reshape and aggregating
        N, nvar, L, E = x.shape
        clss = x[:, :, :1, :]
        x = x[:, :, 1:, :].view(N, nvar, -1, 13, E)
        x = torch.sum(x, dim=3) # N, nvar, L_new, E
        return torch.cat((clss, x), dim=2) # N, nvar, L_new+1, E

    def forward_encoder(self, x):
        '''Input
        x: bs*nvar x 3 x L x F
        
        '''
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: 
        if self.mask_scheme == 'random':
             x, mask, ids_restore = self.random_masking(x,mask_ratio=self.mask_prob)
             
        else:
            #print(self.mask_t_prob,self.mask_f_prob)
            x, mask, ids_restore = self.random_masking_2d(x, mask_t_prob=self.mask_t_prob, mask_f_prob=self.mask_f_prob)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Encoder blocks
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.norm(x) # bs*nvar * p_patches * E

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x) 

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed  

        # decode
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x) # bs*nvar x num_p x 512
        
        # predictor projection
        x = x.permute(0, 2, 1) # bs*nvar x 512 x num_p
        x = self.temporal_recon(x) # bs*nvar x 512 x target_len
        x = self.spatial_recon(x) # bs x nvar x target_len


        return x

    def forward_loss(self,target_tss, pred):
        """
        target_tss: bs x nvar x 388
        pred: bs x nvar x target_len
        """

        # cosim_scores = self.cosim(target_tss,pred)
        # loss = 1 - cosim_scores
        # cos_loss = loss.mean()
        mse_loss = F.mse_loss(pred, target_tss)

        loss = mse_loss
        
        return loss

    def forward(self, imgs, target_tss):
        '''Input
        imgs: bs x nvars x 3 x L x F
        target_tss: bs x nvars x L+1
        '''
        if self.use_cwt: # using cwt
            bs, nvar, ch, L, F = imgs.shape
            imgs = torch.reshape(imgs,(bs*nvar,ch,L,F))
        else: # using raw series
            bs, nvar, L = imgs.shape
            imgs = torch.reshape(imgs,(bs*nvar,L))
        
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)  # bs x nvar x target_len

        loss = self.forward_loss(target_tss, pred)

        return loss, pred, mask

    @torch.no_grad()
    def get_signal_embedding(self, x, hidden_out=False, device=torch.device('cpu')):
        '''Input
        x: bs, nvar, 3, L, F
        '''
        N, nvar, ch, L, F_ = x.shape

        # embed patches
        x = self.patch_embed(torch.reshape(x, (N*nvar, ch, L, F_))) # [N*nvar, P, E]

        # _, new_L, _ = x.shape
        # add pos embed w/o cls token
        pos_embed = self.pos_adjust((L, F_), device=device)
        x = x + pos_embed[:, 1:, :]
        # x = x + self.pos_embed[:, 1:, :]

        # append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Encoder blocks
        hiddens = list()
        for blk in self.encoder_blocks:
            blk.nvar = nvar
            x = blk(x)

            if hidden_out:
                BN, P, E = x.shape
                curr_hidden = torch.reshape(x,(BN//nvar, nvar, P, E))
                if self.comb_freq:
                    curr_hidden = self.combine_freq_patch(curr_hidden)
                hiddens.append(curr_hidden)

        # final transformation
        x = self.norm(x) # bs*nvar * p_patches * E
        BN, P, E = x.shape
        x = torch.reshape(x,(BN//nvar, nvar, P, E))

        # if combining the patches along frequency axis
        if self.comb_freq:
            x = self.combine_freq_patch(x)

        # return
        if hidden_out:
            return x, hiddens
        return x

def ricker_wavelet(points, scale):
    """Generate the Ricker (Mexican hat) wavelet for a given scale."""
    # a = scale
    # A = 2 / (torch.sqrt(3 * a) * torch.pi**0.25)  # Normalization factor
    # wavelet = A * (1 - (t / a)**2) * torch.exp(-0.5 * (t / a)**2)
    # return wavelet

    A = 2 / (torch.sqrt(3 * scale) * torch.pi**0.25)  # Normalization factor
    wsq = scale**2
    vec = torch.arange(0, points, device=scale.device) - (points - 1.0) / 2
    xsq = vec**2
    mod = (1 - xsq / wsq)
    gauss = torch.exp(-xsq / (2 * wsq))
    wavelet = A * mod * gauss
    return wavelet

def cwt_ricker(x, lowest_scale, largest_scale, step=1, wavelet_len=100):
    """
    Compute the CWT using the Ricker wavelet in PyTorch with simplified inputs.
    
    Args:
        x (torch.Tensor): Input time-series of shape (batch_size, sequence_length).
        lowest_scale (float): The lowest scale for the wavelet.
        largest_scale (float): The largest scale for the wavelet.
        step (float): Step size for generating scales.
        wavelet_len (int): Length of the wavelet.
        
    Returns:
        torch.Tensor: CWT scalogram of shape (batch_size, num_scales, sequence_length).
    """
    batch_size, sequence_length = x.shape
    x = x.unsqueeze(1)  # Add channel dimension, now (batch_size, 1, sequence_length)
    
    # Generate scales
    scales = torch.arange(lowest_scale, largest_scale + step, step, device=x.device)
    num_scales = scales.shape[0]
    
    # Prepare the wavelet basis for each scale
    # t = torch.linspace(-wavelet_len // 2, wavelet_len // 2, wavelet_len, device=x.device)
    wavelet_len = min(10*largest_scale, sequence_length)
    wavelets = torch.stack([ricker_wavelet(wavelet_len, scale) for scale in scales])
    # wavelets = torch.stack([ricker_wavelet(min(10*scale, sequence_length), scale) for scale in scales])
    wavelets = wavelets.view(num_scales, 1, -1)  # (num_scales, 1, wavelet_len)
    
    # Perform convolution for each scale
    cwt_output = F.conv1d(x, wavelets, padding=wavelet_len // 2)
    
    return cwt_output

def cwt_wrap(x, lowest_scale, largest_scale, step=1, wavelet_len=100):
    # x: bn, L
    # return: bn, 3, L, n_mels
    d1 = x[:, 1:] - x[:, :-1]  # bn, L-1
    d2 = d1[:, 1:] - d1[:, :-1] # bn, L-2
    x = torch.stack([x[:, 2:], d1[:, 1:], d2]).float().permute(1, 0, 2) # bn, 3, L-1
    bn, n_, new_L = x.shape
    cwt_res = cwt_ricker(x.reshape(bn*n_, new_L), lowest_scale, largest_scale, step=step, wavelet_len=wavelet_len) # bn*3, 65, new_L
    _, n_scale, new_L = cwt_res.shape
    return cwt_res.reshape(bn, n_, n_scale, new_L).permute(0, 1, 3, 2) # bn, 3, L, n_mels

if __name__ == '__main__':
    pass
