from torch import optim
from tqdm import tqdm
import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
import math
from torch.nn import functional as F

from torch.nn.utils import clip_grad_norm_
from pytorch_pretrained_vit import ViT
# from transformer import Block

# from src.models.layers import *

# from layers import FeedForward,VitPosEmbedAdjust,freeze_model,tAPE
import os

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class FeedForward(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout=0.1, add_norm=True, vae_out=False):
        super().__init__()
        self.add_norm = add_norm

        last_out = nn.Linear

        self.fc_liner = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            # nn.Dropout(p=dropout),
            # nn.Linear(hidden_size, emb_size),
            last_out(hidden_size, emb_size),
            nn.Dropout(p=dropout),
        )

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-6)

    def forward(self, x):
        out = self.fc_liner(x)
        if self.add_norm:
            return self.LayerNorm(x + out)
        return out
    
class tAPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x): # N, L, C
        # print(torch.min(self.pe[:, :, :]), torch.max(self.pe[:, :, :])) # [-1, 1]
        # exit()
        x = x + self.pe[:, -x.shape[1]:, :]
        return self.dropout(x)
        
class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))

class MultiHeadedSelfAttentionRaw(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h

class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttentionRaw(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask)))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x
    
# Instead of masking the original cwt, let's try masking on the latents,
# therefore forcing the model use the alternate sequence to learn the missing feature

class CrossViTModel(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4,
        norm_layer = nn.LayerNorm,
        device='cuda',
        max_length = 387,
        decoder_depth=2,
        depth = 12, 
        num_sensor = 11,
        is_test = False, # for visulization
        is_pretrain = True, # for linear prob
        num_class = 1
    ):
        super().__init__()

        self.device = device
        self.is_pretrain = is_pretrain
        self.max_length = max_length
        # for statistics track
        self.stats = dict()
        
        # for visulization
        self.is_test = is_test
        self.cross_attn1 = []
        self.self_attn1 = []
        self.cross_attn2 = []
        self.self_attn2 = []
        # --------------------------------------------------------------------------
        # encoder specifics
        # ViT default patch embeddings

        self.vit = ViT('B_16_imagenet1k', pretrained=True).to(torch.bfloat16).to(self.device) # construct and load 
        self.vit.patch_embedding.stride = (4, 16)
        freeze_model(self.vit)

        self.input_transforms = transforms.Compose([
            # transforms.Resize((L+16, 65)), 
            transforms.Normalize(0.5, 0.5),
            lambda im: nn.functional.pad(im, (0, 0, 16, 0), value=0),
        ]) # N, 3, L, 384

        self.cls_token_1 = nn.Parameter(torch.zeros(1, 1, embed_dim,dtype=torch.bfloat16))
        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, embed_dim,dtype=torch.bfloat16))
        #self.mask_token = nn.Parameter(torch.zeros(1,1,embed_dim,dtype=torch.bfloat16))
        self.pos_embed_1 = tAPE(embed_dim, dropout=0.1, max_len=max_length+1)
        self.pos_embed_2 = tAPE(embed_dim, dropout=0.1, max_len=max_length+1)
        
        # # encoder
        # self.signal1_encoders = nn.ModuleList([
        #     Block(dim=embed_dim, num_heads=num_heads, 
        #           ff_dim=embed_dim*mlp_ratio,
        #           dropout=0.1)
        #     for i in range(2)])
        
        # self.signal2_encoders = nn.ModuleList([
        #     Block(dim=embed_dim, num_heads=num_heads, 
        #           ff_dim=embed_dim*mlp_ratio,
        #           dropout=0.1)
        #     for i in range(2)])

        # multiSignal blocks (self-attn -> cross-attn )
        # self.pos_embed_encode = tAPE(embed_dim, dropout=0.1, max_len=max_length+1)
        
        self.blocks = nn.ModuleList([
                MultiSignalBlock(embed_dim = embed_dim, depth=1, 
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=False, 
                 qk_scale=None, drop=0., attn_drop=0.,
                 norm_layer=norm_layer) for _ in range(depth)
        ])

        self.x1_norm = norm_layer(embed_dim).bfloat16()
        self.x2_norm = norm_layer(embed_dim).bfloat16()

        # Decoder Blocks (light weight)
        # To reconstruct points from the encoder features
        # self.decoder_blocks = nn.ModuleList([
        #     nn.Sequential(
        #         Block(dim=embed_dim, num_heads=num_heads, 
        #           ff_dim=embed_dim*mlp_ratio,
        #           dropout=0.1))
        #     for i in range(decoder_depth)])
        
        # self.decoder_norm = nn.LayerNorm(embed_dim).bfloat16()
        
        #TODO: Head
        self.decoder_pred_1 = nn.Linear(embed_dim, 1).bfloat16() # decoder to series1
        self.decoder_pred_2 = nn.Linear(embed_dim, 1).bfloat16() # decoder to series1

        # if not self.is_pretrain:
        #     self.decoder_pred_1 = nn.ModuleList([nn.Linear(768, num_class).bfloat16() for i in range(1)])
        #     self.decoder_pred_2  = nn.ModuleList([nn.Linear(768, num_class).bfloat16() for i in range(1)])


        self.ar_pred_1 = nn.Linear(embed_dim,1).bfloat16()
        self.ar_pred_2 = nn.Linear(embed_dim,1).bfloat16()

        self.cls_1 = nn.Linear(embed_dim,num_sensor).bfloat16()
        self.cls_2 = nn.Linear(embed_dim,num_sensor).bfloat16()

        # loss
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self,x_1,x_2,
                x1_mask,x2_mask,):
        '''
        @param x1,x2: (N, 3, 387, 65)
        @param padding_mask: (N, 388)        
        '''
        N, _, L, F = x_1.shape
        #print('input shape: ',x_1.shape)
        x_1 = self.input_transforms(x_1)
        x_1 = self.vit.patch_embedding(x_1)
        # print('shape of x_1 patch_embedding,',x_1.shape)
        # print('shape of input padding mask,',x1_mask.shape)
        x_1 = x_1.flatten(2).transpose(1, 2) # b,gh*gw,d
        x_1 = x_1[:, -L:, :] # b,L,d
        
        cls_tokens = self.cls_token_1.expand(x_1.shape[0], -1, -1)
        x_1_input = torch.cat((cls_tokens, x_1), dim=1)
       
        x_1_input = self.pos_embed_1(x_1_input) # N, length * mask_ratio+1, E

        # helper signals
        x_2 = self.input_transforms(x_2)
        x_2 = self.vit.patch_embedding(x_2)
        x_2 = x_2.flatten(2).transpose(1, 2) # b,gh*gw,d
        x_2 = x_2[:, -L:, :] # b,L,d

        cls_tokens_2 = self.cls_token_2.expand(x_2.shape[0], -1, -1)
        x_2_input = torch.cat((cls_tokens_2, x_2), dim=1)
        x_2_input = self.pos_embed_2(x_2_input) # N, num_patches+1,E
        
        # print('shape of x_1_input',x_1_input.shape) # N, 388, 768
        # print('shape of x_2 input: ',x_2_input.shape)
        
        #print('data type of x_1_input',x_1_input.dtype)
        # Self-Attention Encoder, Freeze this two
        # with torch.no_grad():
        #     for b_i in range(4):
        #         x_1_input = self.vit.transformer.blocks[b_i](x_1_input,mask=x1_mask)
        #         x_2_input = self.vit.transformer.blocks[b_i](x_2_input,mask=x2_mask)

        # print('vit feature shape:',x_1_input.shape)
                    
        # Multi-modal Encoder
        if not self.is_test:
            for blk in self.blocks:
                x_1_input,x_2_input = blk(x_1 = x_1_input,
                                        x_2 = x_2_input,
                                        x1_attn=x1_mask,
                                        x2_attn=x2_mask,
                                        is_test = self.is_test)
         # For Visulization ###########################################
        if self.is_test:
            for blk in self.blocks:
                x_1_input,x_2_input,corss_attn_matrix1,\
                cross_attn_matrix2,self_attn_matrix1,\
                self_attn_matrix2 = blk(x_1 = x_1_input,
                                        x_2 = x_2_input,
                                        x1_attn=x1_mask,
                                        x2_attn=x2_mask,
                                        is_test = self.is_test)
                self.cross_attn1.append(corss_attn_matrix1)
                self.cross_attn2.append(cross_attn_matrix2)
                self.self_attn1.append(self_attn_matrix1)
                self.self_attn2.append(self_attn_matrix2)
        ###############################################################   
            
        x1_out = self.x1_norm(x_1_input) # N, num_patches+1,E
        x2_out = self.x2_norm(x_2_input)


        # # Decoder
        # for blk in self.decoder_blocks:
        #     x_out = blk(x_out,mask=x1_mask)

        # x_out = self.decoder_norm(x_out)
        # cls_out = x_out[:,0,:] 


        if self.is_pretrain:# Reconstruct
            reconstruct_x1 = self.decoder_pred_1(x1_out[:,1:,:]).squeeze(2) # N, 387
            reconstruct_x2 = self.decoder_pred_2(x2_out[:,1:,:]).squeeze(2) # N, 387

            # TODO: Next_value Prediction
            ar_x1 = self.ar_pred_1(x1_out[:,0,:]).squeeze(1)
            ar_x2 = self.ar_pred_2(x1_out[:,0,:]).squeeze(1)

            sensor_cls_x1 = self.cls_1(x1_out[:,0,:])
            sensor_cls_x2 = self.cls_2(x2_out[:,0,:])

            return x1_out,x2_out,reconstruct_x1,reconstruct_x2,ar_x1,ar_x2,sensor_cls_x1,sensor_cls_x2
        
        else: 
            #hack: we will replace decoder_pred_1, and decoder_pred_2 with a module_list for linear probing
            result_list = []
            for linear1,linear2 in zip(self.decoder_pred_1,self.decoder_pred_2):
                x1_out = linear1(torch.mean(x1_out[:,1:,:],dim=1)) # N x num_classes
                x2_out = linear2(torch.mean(x2_out[:,1:,:],dim=1)) # N x num_classes
                result = (x1_out+x2_out)/2 # N x num_classes
                result_list.append(result) # Address the situation where there are multiple task

            return result_list #  num_tasks x tensor(N x num_classes) 

    
    def loss_f(self, pred_ts1, pred_ts2,
           ar_x1, ar_x2,
           target_ts1, target_ts2,
           sensor_cls_x1, sensor_cls_x2,
           target_cls1, target_cls2,
           x1_mask, x2_mask):
        
        '''
        pred_ts1, pred_ts2: N, 387
        '''

        # Extract the last timestep for autoregression loss
        target_ar1 = target_ts1[:, -1]
        target_ar2 = target_ts2[:, -1]

        # Calculate reconstruction loss
        re_loss1 = torch.sum((pred_ts1 - target_ts1[:, :-1]) ** 2 * x1_mask[:, :-1]) / torch.sum(x1_mask[:, :-1])
        re_loss2 = torch.sum((pred_ts2 - target_ts2[:, :-1]) ** 2 * x2_mask[:, 1:]) / torch.sum(x2_mask[:, 1:])

        # Calculate L1 loss for each branch
        l1_loss1 = self.l1_loss(ar_x1, target_ar1)
        l1_loss2 = self.l1_loss(ar_x2, target_ar2)

        # Calculate cross-entropy loss for each branch
        ce_loss1 = self.ce_loss(sensor_cls_x1, target_cls1)
        ce_loss2 = self.ce_loss(sensor_cls_x2, target_cls2)

        # Calculate total losses for each branch
        loss_1 = 0.4 * re_loss1 + 0.3 * l1_loss1 + 0.3 * ce_loss1
        loss_2 = 0.4 * re_loss2 + 0.3 * l1_loss2 + 0.3 * ce_loss2

        # Total losses
        total_loss = loss_1 + loss_2

        return total_loss, {'re_loss1': re_loss1.item(), 're_loss2': re_loss2.item(),
                            'l1_loss1': l1_loss1.item(), 'l1_loss2': l1_loss2.item(),
                            'ce_loss1': ce_loss1.item(), 'ce_loss2': ce_loss2.item()}

    
    @torch.no_grad()
    def forward_all(self,all_cwt):
        # Input: all_cwt shape: tensor(N x ch x 3 x L x 65)
        # Output: all_cwt embedding: tensor(N x ch x L x 768)

        N,ch,_,L,F = all_cwt.shape
        #print('**cwt ch: ',ch)
        all_signal_pairs = torch.empty(N,0,2,3,self.max_length,F).to(self.device)
        all_mask_pairs = torch.empty(N,0,2,388).to(self.device)
        pair_index_record = torch.empty(N,0,2)
        if ch == 1:
            x_1 = all_cwt.squeeze(1)  # N x 3 x L x 65
            x_1,padding_mask_1 = self.padding(x_1)
            x_2 = x_1.clone()
            padding_mask_2 = padding_mask_1.clone() # N x L+1

            print('shape of inference input sample: ',x_1.shape) #  N x 3 x L x 65
            pair_signal = torch.cat((x_1.unsqueeze(1), x_2.unsqueeze(1)), dim=1) # N x 2 x 3 x L x 65
            pair_mask =  torch.cat((padding_mask_1.unsqueeze(1), padding_mask_2.unsqueeze(1)), dim=1) # N x 2 x 388

            all_signal_pairs = torch.cat((all_signal_pairs,pair_signal.unsqueeze(1)),dim=1) # N x 1 x 2 x 3 x 387 x 65
            all_mask_pairs = torch.cat((all_mask_pairs,pair_mask.unsqueeze(1)),dim=1) # N x 1 x 2 x 388
            pair_index_record = torch.cat((pair_index_record,torch.tensor([0,0]).repeat(N, 1).unsqueeze(1)),dim=1) # N x 1 x 2
        

        #print('**CH: ',ch)
        for i in range(ch):
            for j in range(i+1,ch):
                x_1 = all_cwt[:,i,...] # N x 3 x L x 65
                x_2 = all_cwt[:,j,...] # N x 3 x L x 65

                x_1,padding_mask_1 = self.padding(x_1)
                x_2,padding_mask_2 = self.padding(x_2)
                
                pair_signal = torch.cat((x_1.unsqueeze(1), x_2.unsqueeze(1)), dim=1) # N x 2 x 3 x 387 x 65
                pair_mask =  torch.cat((padding_mask_1.unsqueeze(1), padding_mask_2.unsqueeze(1)), dim=1) # N x 2 x 388

                all_signal_pairs = torch.cat((all_signal_pairs,pair_signal.unsqueeze(1)),dim=1) # N x num_pairs x 2 x 3 x 387 x 65
                all_mask_pairs = torch.cat((all_mask_pairs,pair_mask.unsqueeze(1)),dim=1) # N x num_pairs x 2 x 388
                pair_index_record = torch.cat((pair_index_record,torch.tensor([i,j]).repeat(N, 1).unsqueeze(1)),dim=1) # N x num_pairs x 2

                #print('all_signal_pairs shape: ',all_signal_pairs.shape)
        N, num_pairs,_,_,L,F = all_signal_pairs.shape
        print('all signals pair shape:',all_signal_pairs.shape)

        reshaped_signal_pairs = all_signal_pairs.reshape(N * num_pairs, 2, 3, L, F)
        reshaped_mask_pairs = all_mask_pairs.reshape(N * num_pairs, 2, -1)
        reshaped_pair_record = pair_index_record.reshape(N * num_pairs, 2).to(self.device)

        # print('shape of reshaped_signal_pairs',reshaped_signal_pairs.shape)
        # print('shape of reshaped_mask_pairs',reshaped_mask_pairs.shape)
        # print('shape of reshaped_pair_record',reshaped_pair_record.shape)

        print('Start Inference: ')
        
        x_1 = reshaped_signal_pairs[:,0,...].bfloat16() # (N x # of pairs) x 3 x 387 x 65
        x_2 = reshaped_signal_pairs[:,1,...].bfloat16() # (N x # of pairs) x 3 x 387 x 65

        x_1_attn_mask = reshaped_mask_pairs[:,0,...].bfloat16() # (N x # of pairs) x 388
        x_2_attn_mask = reshaped_mask_pairs[:,1,...].bfloat16() # (N x # of pairs) x 388

        x1_out,x2_out,_,_,_,_,_,_ = self.forward(x_1,x_2,
                            x1_mask=x_1_attn_mask,
                            x2_mask=x_2_attn_mask)
        
        # x1_out shape: (Nxnum_pairs) x (L+1) x E
        # x2_out shape: (Nxnum_pairs) x (L+1) x E

        print('x1_out shape: ',x1_out.shape)
        print('x2_out shape: ',x2_out.shape)
        
        combine_tensor = torch.cat((x1_out.unsqueeze(1),x2_out.unsqueeze(1)),dim=1) # (Nxnum_pairs) x 2 x (L+1) x E
        combine_tensor = torch.cat((x1_out.unsqueeze(1),x2_out.unsqueeze(1)),dim=1) # (Nxnum_pairs) x 2 x (L+1) x E

        embed_out = []
        for i in range(0,N*num_pairs,num_pairs):
            curr_batch = combine_tensor[i:i+num_pairs,...]
            curr_idx_record = reshaped_pair_record[i:i+num_pairs,...]

            batch_embed = torch.empty(0,L+1,768).to(self.device)
            for i in range(ch):
                selected_tensor = curr_batch[curr_idx_record==i] # num x (L+1) x E
                #print('shape of selected_tensor: ',selected_tensor.shape)
                if selected_tensor.numel() != 0:
                    mean_tensor = selected_tensor.mean(dim=0) # L+1 x E
                    batch_embed = torch.cat((batch_embed,mean_tensor.unsqueeze(0)),dim=0) # num_ch x L+1 x E

            embed_out.append(batch_embed.unsqueeze(0)) # # N x num_ch x L+1 x 768
    
        embed_out = torch.cat(embed_out, dim=0).to(self.device) # N x num_ch x L+1 x 768

        print('Embedding out shape',embed_out.shape)
        
        return embed_out
    

    def padding(self,in_cwt):
        N,_,L,F = in_cwt.shape
        # reshape to C,L,F
        max_length = self.max_length+1

        cwt_length = L
        ts_length = cwt_length+1
            
        padding_mask = torch.ones(N,self.max_length+1).to(self.device) # N, 388
        
        if ts_length > max_length:
            in_cwt = in_cwt[:, :,-max_length+1:, :] #3,max_length,65

        elif ts_length < max_length:
            in_cwt = torch.cat((torch.zeros(N,3,max_length-1-L, F).to(self.device), in_cwt), dim=2) # 3, max_len, 65
            padding_mask[:,:self.max_length-L+1] = 0

        return in_cwt, padding_mask



    def num_params(self):
        total_num = sum(p.numel() for p in self.parameters())
        train_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total of Parameters: {}M".format(round(total_num / 1e6, 2)))
        print("Train Parameters: {}M".format(round(train_num / 1e6, 2)))
    
    

    def fit(
        self,
        dataloader,
        epochs,
        lr=1e-3,
        weight_decay=1e-2
    ):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # train
        train_losses = list()
        print("Start training")
        for e in tqdm(range(epochs)):
            if len(train_losses) > 0:
                print("Last Train Loss:", train_losses[-1])
            self.train()
            for data in tqdm(dataloader):
                # forward
                x1_out,x2_out,reconstruct_x1,reconstruct_x2,ar_x1,ar_x2,sensor_cls_x1,sensor_cls_x2 = self(
                    x_1=data["sample_in_1"],
                    x_2=data['sample_in_2'],
                    x1_mask = data['padding_mask_1'],
                    x2_mask = data['padding_mask_2'],
                )

                loss,loss_dict = self.loss_f(pred_ts1=reconstruct_x1,
                                   pred_ts2=reconstruct_x2,
                                   ar_x1 = ar_x1,
                                   ar_x2 = ar_x2,
                                   target_ts1=data['sample_out_1'],
                                   target_ts2=data['sample_out_2'],
                                   sensor_cls_x1 = sensor_cls_x1,
                                   sensor_cls_x2 = sensor_cls_x2,
                                   target_cls1 = data['cls_1'],
                                   target_cls2 = data['cls_2'],
                                   x1_mask=data['padding_mask_1'],
                                   x2_mask = data['padding_mask_2'])
                

                # backprop
                optimizer.zero_grad() # clear cache
                loss.backward() # calculate gradient

                # gradient clipping
                for p in self.parameters(): # addressing gradient vanishing
                    if p.requires_grad and p.grad is not None:
                        p.grad = torch.nan_to_num(p.grad, nan=0.0)
                clip_grad_norm_(self.parameters(), 5)

                optimizer.step() # update parameters

                # update record
                train_losses.append(loss.detach().cpu().item())
                for key,value in loss_dict.items():
                    print(key,value)

                # for i in range(len(data["in_tk"])):
                #     in_tk, out_tk = data["in_tk"][i], data["out_tk"][i]
                #     comb_tk = "{}-{}".format(in_tk, out_tk)
                #     if self.stats.get(in_tk) is None:
                #         self.stats[in_tk] = 0
                #     if self.stats.get(out_tk) is None:
                #         self.stats[out_tk] = 0
                #     if self.stats.get(comb_tk) is None:
                #         self.stats[comb_tk] = 0
                #     self.stats[in_tk] += 1
                #     self.stats[out_tk] += 1
                #     self.stats[comb_tk] += 1


        self.eval()

        return train_losses

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        # cross attention of q: cls token and kv: whole sequence

        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias).bfloat16()
        self.wk = nn.Linear(dim, dim, bias=qkv_bias).bfloat16()
        self.wv = nn.Linear(dim, dim, bias=qkv_bias).bfloat16()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim).bfloat16()
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None,is_test=False): 
        # x = (x_1-cls token, x_2 tokens)
        B, N, C = x.shape #bs, num_patches+1, E, 
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        if mask is not None:
            mask = mask[:,None,None,:].bfloat16()
            attn -= 1000.0*(1.0-mask)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)

        if is_test: return x,attn

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm, has_mlp=False):
        super().__init__()
        self.norm1 = norm_layer(dim).bfloat16()
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, 
            proj_drop=drop)
        
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim).bfloat16()
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = FeedForward(emb_size=dim, hidden_size=mlp_hidden_dim)

    def forward(self, x,mask,is_test=False):
        if is_test: tmp,attn_matrix = self.attn(self.norm1(x),mask,is_test)
        else: tmp = self.attn(self.norm1(x),mask,is_test)

        x = x[:, 0:1, ...] + tmp
        
        if self.has_mlp:
            x = x + self.mlp(self.norm2(x))

        if is_test: return x, attn_matrix

        return x # N, 1, E
    

class MultiSignalBlock(nn.Module):
    def __init__(self,embed_dim = 768, depth=1, 
                 num_heads=12, mlp_ratio=2.0, qkv_bias=False, 
                 qk_scale=None, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.signal1_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, 
                  ff_dim=embed_dim*mlp_ratio,
                  dropout=0.1)
            for i in range(depth)])
        

        self.signal2_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, 
                  ff_dim=embed_dim*mlp_ratio,
                  dropout=0.1)
            for i in range(depth)])

        
        # self.cls_proj = FeedForward(emb_size=embed_dim,hidden_size=embed_dim*mlp_ratio)
        # self.revert_cls_proj = FeedForward(emb_size=embed_dim,hidden_size=embed_dim*mlp_ratio)

        self.x_1fusion = CrossAttentionBlock(dim=embed_dim,num_heads=num_heads,
                                          mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,
                                          qk_scale=qk_scale,drop=drop,attn_drop=attn_drop,
                                          norm_layer=norm_layer,
                                          has_mlp=False)
        
        self.x_2fusion = CrossAttentionBlock(dim=embed_dim,num_heads=num_heads,
                                          mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,
                                          qk_scale=qk_scale,drop=drop,attn_drop=attn_drop,
                                          norm_layer=norm_layer,
                                          has_mlp=False)
        
    def forward(self,x_1,x_2,
                x1_attn=None,x2_attn=None,is_test=False):
        
        # Self-Attention Block
        for blk in self.signal1_blocks:
            x_1 = blk(x_1,mask=x1_attn) # N, num_patches+1, E
            
        for blk in self.signal2_blocks:
            x_2 = blk(x_2,mask=x2_attn) # N, num_patches+1, E

        if is_test:
            self_attn_matrix1 = self.signal1_blocks[0].attn.scores
            self_attn_matrix2 = self.signal2_blocks[0].attn.scores
            
        ##################################################################################
        x1_cls = x_1[:, 0:1,:]
        x1_ts = x_1[:, 1:,:]
        x2_cls = x_2[:, 0:1,:]
        x2_ts = x_2[:, 1:,:]
        ##################################################################################
        # x_1 branch
        # take x_1 cls_token out and concat with x_2 tokens

        # x_in = torch.cat((self.cls_proj(cls_1),x_2[:,1:,:]),dim=1) # N, num_patches+1, E
        x1_kqv = torch.cat((x1_cls,x2_ts),dim=1)
        if is_test: tmp1,corss_attn_matrix1 = self.x_1fusion(x1_kqv,x1_attn,is_test) # N, 1, E
        else: tmp1=self.x_1fusion(x1_kqv,x1_attn,is_test)
        
        #print('dimension of cross-attention token: ',tmp.shape)
        #cls_out = self.revert_cls_proj(tmp)

        # concat the fuse-clse token with the original x_1 tokens
        x_1_out = torch.cat((tmp1,x1_ts),dim=1) # N, num_patches+1, E

        ##################################################################################
        # TODO: x_2 branch
        x2_kqv = torch.cat((x2_cls,x1_ts),dim=1)
        if is_test: tmp2,cross_attn_matrix2 = self.x_2fusion(x2_kqv,x2_attn,is_test)
        else: tmp2 = self.x_2fusion(x2_kqv,x2_attn,is_test)
        x_2_out = torch.cat((tmp2,x2_ts),dim=1)
        ##################################################################################

        if is_test: return x_1_out,x_2_out,corss_attn_matrix1,cross_attn_matrix2,self_attn_matrix1,self_attn_matrix2

        return x_1_out,x_2_out
    


class CrossViTBlock(nn.Module):
    def __init__(self,embed_dim = 768, depth=1,self_attn_model = None, 
                 num_heads=12, mlp_ratio=2.0, qkv_bias=False, 
                 qk_scale=None, drop=0., attn_drop=0.,
                 curr_layer = 0,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.self_attn = self_attn_model.transformer.blocks[curr_layer].eval()

        # self.cls_proj = FeedForward(emb_size=embed_dim,hidden_size=embed_dim*mlp_ratio)
        # self.revert_cls_proj = FeedForward(emb_size=embed_dim,hidden_size=embed_dim*mlp_ratio)

        self.x_1fusion = CrossAttentionBlock(dim=embed_dim,num_heads=num_heads,
                                          mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,
                                          qk_scale=qk_scale,drop=drop,attn_drop=attn_drop,
                                          norm_layer=norm_layer,
                                          has_mlp=False)
        
        self.x_2fusion = CrossAttentionBlock(dim=embed_dim,num_heads=num_heads,
                                          mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,
                                          qk_scale=qk_scale,drop=drop,attn_drop=attn_drop,
                                          norm_layer=norm_layer,
                                          has_mlp=False)
        
    def forward(self,x_1,x_2,
                x1_attn=None,x2_attn=None,is_test=False):
        
        # Self-Attention Block
        x_1 = self.self_attn(x_1,mask=x1_attn) # N, num_patches+1, E
        x_2 = self.self_attn(x_2,mask=x2_attn) # N, num_patches+1, E
            

        # if is_test:
        #     self_attn_matrix1 = self.signal1_blocks[0].attn.scores
        #     self_attn_matrix2 = self.signal2_blocks[0].attn.scores
            
        ##################################################################################
        x1_cls = x_1[:, 0:1,:]
        x1_ts = x_1[:, 1:,:]
        x2_cls = x_2[:, 0:1,:]
        x2_ts = x_2[:, 1:,:]
        ##################################################################################
        # x_1 branch
        # take x_1 cls_token out and concat with x_2 tokens

        # x_in = torch.cat((self.cls_proj(cls_1),x_2[:,1:,:]),dim=1) # N, num_patches+1, E
        x1_kqv = torch.cat((x1_cls,x2_ts),dim=1)
        if is_test: tmp1,corss_attn_matrix1 = self.x_1fusion(x1_kqv,x1_attn,is_test) # N, 1, E
        else: tmp1=self.x_1fusion(x1_kqv,x1_attn,is_test)
        
        #print('dimension of cross-attention token: ',tmp.shape)
        #cls_out = self.revert_cls_proj(tmp)

        # concat the fuse-clse token with the original x_1 tokens
        x_1_out = torch.cat((tmp1,x1_ts),dim=1) # N, num_patches+1, E

        ##################################################################################
        # TODO: x_2 branch
        x2_kqv = torch.cat((x2_cls,x1_ts),dim=1)
        if is_test: tmp2,cross_attn_matrix2 = self.x_2fusion(x2_kqv,x2_attn,is_test)
        else: tmp2 = self.x_2fusion(x2_kqv,x2_attn,is_test)
        x_2_out = torch.cat((tmp2,x2_ts),dim=1)
        ##################################################################################

        #if is_test: return x_1_out,x_2_out,corss_attn_matrix1,cross_attn_matrix2,self_attn_matrix1,self_attn_matrix2

        return x_1_out,x_2_out
    



class CrossSignalViT(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4,
        norm_layer = nn.LayerNorm,
        device='cuda',
        max_length = 387,
        decoder_depth=2,
        depth = 12, 
        num_sensor = 11,
        is_test = False, # for visulization
        is_pretrain = True, # for linear prob
        num_class = 1
    ):
        super().__init__()

        self.device = device
        self.is_pretrain = is_pretrain
        self.max_length = max_length
        # for statistics track
        self.stats = dict()
        
        # for visulization
        self.is_test = is_test
        self.cross_attn1 = []
        self.self_attn1 = []
        self.cross_attn2 = []
        self.self_attn2 = []
        # --------------------------------------------------------------------------
        # encoder specifics
        # ViT default patch embeddings

        self.vit = ViT('B_16_imagenet1k', pretrained=True).to(torch.bfloat16).to(self.device) # construct and load 
        self.vit.patch_embedding.stride = (4, 16)
        freeze_model(self.vit)

        self.input_transforms = transforms.Compose([
            # transforms.Resize((L+16, 65)), 
            transforms.Normalize(0.5, 0.5),
            lambda im: nn.functional.pad(im, (0, 0, 16, 0), value=0),
        ]) # N, 3, L, 384

        self.cls_token_1 = nn.Parameter(torch.zeros(1, 1, embed_dim,dtype=torch.bfloat16))
        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, embed_dim,dtype=torch.bfloat16))
        #self.mask_token = nn.Parameter(torch.zeros(1,1,embed_dim,dtype=torch.bfloat16))
        self.pos_embed_1 = tAPE(embed_dim, dropout=0.1, max_len=max_length+1)
        self.pos_embed_2 = tAPE(embed_dim, dropout=0.1, max_len=max_length+1)
        
        # # encoder
        # self.signal1_encoders = nn.ModuleList([
        #     Block(dim=embed_dim, num_heads=num_heads, 
        #           ff_dim=embed_dim*mlp_ratio,
        #           dropout=0.1)
        #     for i in range(2)])
        
        # self.signal2_encoders = nn.ModuleList([
        #     Block(dim=embed_dim, num_heads=num_heads, 
        #           ff_dim=embed_dim*mlp_ratio,
        #           dropout=0.1)
        #     for i in range(2)])

        # multiSignal blocks (self-attn -> cross-attn )
        # self.pos_embed_encode = tAPE(embed_dim, dropout=0.1, max_len=max_length+1)
        
        self.blocks = nn.ModuleList([
                CrossViTBlock(embed_dim = embed_dim, depth=1, 
                              curr_layer = curr_layer,self_attn_model = self.vit,
                              num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=False, 
                              qk_scale=None, drop=0., attn_drop=0.,
                              norm_layer=norm_layer) for curr_layer in range(depth)])

        self.x1_norm = norm_layer(embed_dim).bfloat16()
        self.x2_norm = norm_layer(embed_dim).bfloat16()

        # Decoder Blocks (light weight)
        # To reconstruct points from the encoder features
        # self.decoder_blocks = nn.ModuleList([
        #     nn.Sequential(
        #         Block(dim=embed_dim, num_heads=num_heads, 
        #           ff_dim=embed_dim*mlp_ratio,
        #           dropout=0.1))
        #     for i in range(decoder_depth)])
        
        # self.decoder_norm = nn.LayerNorm(embed_dim).bfloat16()
        
        #TODO: Head
        self.decoder_pred_1 = nn.Linear(embed_dim, 1).bfloat16() # decoder to series1
        self.decoder_pred_2 = nn.Linear(embed_dim, 1).bfloat16() # decoder to series1

        # if not self.is_pretrain:
        #     self.decoder_pred_1 = nn.ModuleList([nn.Linear(768, num_class).bfloat16() for i in range(1)])
        #     self.decoder_pred_2  = nn.ModuleList([nn.Linear(768, num_class).bfloat16() for i in range(1)])


        self.ar_pred_1 = nn.Linear(embed_dim,1).bfloat16()
        self.ar_pred_2 = nn.Linear(embed_dim,1).bfloat16()

        self.cls_1 = nn.Linear(embed_dim,num_sensor).bfloat16()
        self.cls_2 = nn.Linear(embed_dim,num_sensor).bfloat16()

        # loss
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self,x_1,x_2,
                x1_mask,x2_mask,):
        '''
        @param x1,x2: (N, 3, 387, 65)
        @param padding_mask: (N, 388)        
        '''
        N, _, L, F = x_1.shape
        #print('input shape: ',x_1.shape)
        x_1 = self.input_transforms(x_1)
        x_1 = self.vit.patch_embedding(x_1)
        # print('shape of x_1 patch_embedding,',x_1.shape)
        # print('shape of input padding mask,',x1_mask.shape)
        x_1 = x_1.flatten(2).transpose(1, 2) # b,gh*gw,d
        x_1 = x_1[:, -L:, :] # b,L,d
        
        cls_tokens = self.cls_token_1.expand(x_1.shape[0], -1, -1)
        x_1_input = torch.cat((cls_tokens, x_1), dim=1)
       
        x_1_input = self.pos_embed_1(x_1_input) # N, length * mask_ratio+1, E

        # helper signals
        x_2 = self.input_transforms(x_2)
        x_2 = self.vit.patch_embedding(x_2)
        x_2 = x_2.flatten(2).transpose(1, 2) # b,gh*gw,d
        x_2 = x_2[:, -L:, :] # b,L,d

        cls_tokens_2 = self.cls_token_2.expand(x_2.shape[0], -1, -1)
        x_2_input = torch.cat((cls_tokens_2, x_2), dim=1)
        x_2_input = self.pos_embed_2(x_2_input) # N, num_patches+1,E
        
        # print('shape of x_1_input',x_1_input.shape) # N, 388, 768
        # print('shape of x_2 input: ',x_2_input.shape)
        
        #print('data type of x_1_input',x_1_input.dtype)
        # Self-Attention Encoder, Freeze this two
        # with torch.no_grad():
        #     for b_i in range(4):
        #         x_1_input = self.vit.transformer.blocks[b_i](x_1_input,mask=x1_mask)
        #         x_2_input = self.vit.transformer.blocks[b_i](x_2_input,mask=x2_mask)

        # print('vit feature shape:',x_1_input.shape)
                    
        # Multi-modal Encoder
        if not self.is_test:
            for blk in self.blocks:
                x_1_input,x_2_input = blk(x_1 = x_1_input,
                                        x_2 = x_2_input,
                                        x1_attn=x1_mask,
                                        x2_attn=x2_mask,
                                        is_test = self.is_test)
         # For Visulization ###########################################
        if self.is_test:
            for blk in self.blocks:
                x_1_input,x_2_input,corss_attn_matrix1,\
                cross_attn_matrix2,self_attn_matrix1,\
                self_attn_matrix2 = blk(x_1 = x_1_input,
                                        x_2 = x_2_input,
                                        x1_attn=x1_mask,
                                        x2_attn=x2_mask,
                                        is_test = self.is_test)
                self.cross_attn1.append(corss_attn_matrix1)
                self.cross_attn2.append(cross_attn_matrix2)
                self.self_attn1.append(self_attn_matrix1)
                self.self_attn2.append(self_attn_matrix2)
        ###############################################################   
            
        x1_out = self.x1_norm(x_1_input) # N, num_patches+1,E
        x2_out = self.x2_norm(x_2_input)


        # # Decoder
        # for blk in self.decoder_blocks:
        #     x_out = blk(x_out,mask=x1_mask)

        # x_out = self.decoder_norm(x_out)
        # cls_out = x_out[:,0,:] 


        if self.is_pretrain:# Reconstruct
            reconstruct_x1 = self.decoder_pred_1(x1_out[:,1:,:]).squeeze(2) # N, 387
            reconstruct_x2 = self.decoder_pred_2(x2_out[:,1:,:]).squeeze(2) # N, 387

            # TODO: Next_value Prediction
            ar_x1 = self.ar_pred_1(x1_out[:,0,:]).squeeze(1)
            ar_x2 = self.ar_pred_2(x1_out[:,0,:]).squeeze(1)

            sensor_cls_x1 = self.cls_1(x1_out[:,0,:])
            sensor_cls_x2 = self.cls_2(x2_out[:,0,:])

            return x1_out,x2_out,reconstruct_x1,reconstruct_x2,ar_x1,ar_x2,sensor_cls_x1,sensor_cls_x2
        
        else: 
            #hack: we will replace decoder_pred_1, and decoder_pred_2 with a module_list for linear probing
            result_list = []
            for linear1,linear2 in zip(self.decoder_pred_1,self.decoder_pred_2):
                x1_out = linear1(torch.mean(x1_out[:,1:,:],dim=1)) # N x num_classes
                x2_out = linear2(torch.mean(x2_out[:,1:,:],dim=1)) # N x num_classes
                result = (x1_out+x2_out)/2 # N x num_classes
                result_list.append(result) # Address the situation where there are multiple task

            return result_list #  num_tasks x tensor(N x num_classes) 
        

    def loss_f(self, pred_ts1, pred_ts2,
           ar_x1, ar_x2,
           target_ts1, target_ts2,
           sensor_cls_x1, sensor_cls_x2,
           target_cls1, target_cls2,
           x1_mask, x2_mask):
        
        '''
        pred_ts1, pred_ts2: N, 387
        '''

        # Extract the last timestep for autoregression loss
        target_ar1 = target_ts1[:, -1]
        target_ar2 = target_ts2[:, -1]

        # Calculate reconstruction loss
        re_loss1 = torch.sum((pred_ts1 - target_ts1[:, :-1]) ** 2 * x1_mask[:, :-1]) / torch.sum(x1_mask[:, :-1])
        re_loss2 = torch.sum((pred_ts2 - target_ts2[:, :-1]) ** 2 * x2_mask[:, 1:]) / torch.sum(x2_mask[:, 1:])

        # Calculate L1 loss for each branch
        l1_loss1 = self.l1_loss(ar_x1, target_ar1)
        l1_loss2 = self.l1_loss(ar_x2, target_ar2)

        # Calculate cross-entropy loss for each branch
        ce_loss1 = self.ce_loss(sensor_cls_x1, target_cls1)
        ce_loss2 = self.ce_loss(sensor_cls_x2, target_cls2)

        # Calculate total losses for each branch
        loss_1 = 0.4 * re_loss1 + 0.3 * l1_loss1 + 0.3 * ce_loss1
        loss_2 = 0.4 * re_loss2 + 0.3 * l1_loss2 + 0.3 * ce_loss2

        # Total losses
        total_loss = loss_1 + loss_2

        return total_loss, {'re_loss1': re_loss1.item(), 're_loss2': re_loss2.item(),
                            'l1_loss1': l1_loss1.item(), 'l1_loss2': l1_loss2.item(),
                            'ce_loss1': ce_loss1.item(), 'ce_loss2': ce_loss2.item()}

    
    @torch.no_grad()
    def forward_all(self,all_cwt):
        # Input: all_cwt shape: tensor(N x ch x 3 x L x 65)
        # Output: all_cwt embedding: tensor(N x ch x L x 768)

        N,ch,_,L,F = all_cwt.shape
        #print('**cwt ch: ',ch)
        all_signal_pairs = torch.empty(N,0,2,3,self.max_length,F).to(self.device)
        all_mask_pairs = torch.empty(N,0,2,388).to(self.device)
        pair_index_record = torch.empty(N,0,2)
        if ch == 1:
            x_1 = all_cwt.squeeze(1)  # N x 3 x L x 65
            x_1,padding_mask_1 = self.padding(x_1)
            x_2 = x_1.clone()
            padding_mask_2 = padding_mask_1.clone() # N x L+1

            # print('shape of inference input sample: ',x_1.shape) #  N x 3 x L x 65
            pair_signal = torch.cat((x_1.unsqueeze(1), x_2.unsqueeze(1)), dim=1) # N x 2 x 3 x L x 65
            pair_mask =  torch.cat((padding_mask_1.unsqueeze(1), padding_mask_2.unsqueeze(1)), dim=1) # N x 2 x 388

            all_signal_pairs = torch.cat((all_signal_pairs,pair_signal.unsqueeze(1)),dim=1) # N x 1 x 2 x 3 x 387 x 65
            all_mask_pairs = torch.cat((all_mask_pairs,pair_mask.unsqueeze(1)),dim=1) # N x 1 x 2 x 388
            pair_index_record = torch.cat((pair_index_record,torch.tensor([0,0]).repeat(N, 1).unsqueeze(1)),dim=1) # N x 1 x 2
        

        #print('**CH: ',ch)
        for i in range(ch):
            for j in range(i+1,ch):
                x_1 = all_cwt[:,i,...] # N x 3 x L x 65
                x_2 = all_cwt[:,j,...] # N x 3 x L x 65

                x_1,padding_mask_1 = self.padding(x_1)
                x_2,padding_mask_2 = self.padding(x_2)
                
                pair_signal = torch.cat((x_1.unsqueeze(1), x_2.unsqueeze(1)), dim=1) # N x 2 x 3 x 387 x 65
                pair_mask =  torch.cat((padding_mask_1.unsqueeze(1), padding_mask_2.unsqueeze(1)), dim=1) # N x 2 x 388

                all_signal_pairs = torch.cat((all_signal_pairs,pair_signal.unsqueeze(1)),dim=1) # N x num_pairs x 2 x 3 x 387 x 65
                all_mask_pairs = torch.cat((all_mask_pairs,pair_mask.unsqueeze(1)),dim=1) # N x num_pairs x 2 x 388
                pair_index_record = torch.cat((pair_index_record,torch.tensor([i,j]).repeat(N, 1).unsqueeze(1)),dim=1) # N x num_pairs x 2

                #print('all_signal_pairs shape: ',all_signal_pairs.shape)
        N, num_pairs,_,_,L,F = all_signal_pairs.shape
        # print('all signals pair shape:',all_signal_pairs.shape)

        reshaped_signal_pairs = all_signal_pairs.reshape(N * num_pairs, 2, 3, L, F)
        reshaped_mask_pairs = all_mask_pairs.reshape(N * num_pairs, 2, -1)
        reshaped_pair_record = pair_index_record.reshape(N * num_pairs, 2).to(self.device)

        # print('shape of reshaped_signal_pairs',reshaped_signal_pairs.shape)
        # print('shape of reshaped_mask_pairs',reshaped_mask_pairs.shape)
        # print('shape of reshaped_pair_record',reshaped_pair_record.shape)

        # print('Start Inference: ')
        
        x_1 = reshaped_signal_pairs[:,0,...].bfloat16() # (N x # of pairs) x 3 x 387 x 65
        x_2 = reshaped_signal_pairs[:,1,...].bfloat16() # (N x # of pairs) x 3 x 387 x 65

        x_1_attn_mask = reshaped_mask_pairs[:,0,...].bfloat16() # (N x # of pairs) x 388
        x_2_attn_mask = reshaped_mask_pairs[:,1,...].bfloat16() # (N x # of pairs) x 388

        x1_out,x2_out,_,_,_,_,_,_ = self.forward(x_1,x_2,
                            x1_mask=x_1_attn_mask,
                            x2_mask=x_2_attn_mask)
        
        # x1_out shape: (Nxnum_pairs) x (L+1) x E
        # x2_out shape: (Nxnum_pairs) x (L+1) x E

        # print('x1_out shape: ',x1_out.shape)
        # print('x2_out shape: ',x2_out.shape)
        
        combine_tensor = torch.cat((x1_out.unsqueeze(1),x2_out.unsqueeze(1)),dim=1) # (Nxnum_pairs) x 2 x (L+1) x E
        combine_tensor = torch.cat((x1_out.unsqueeze(1),x2_out.unsqueeze(1)),dim=1) # (Nxnum_pairs) x 2 x (L+1) x E

        embed_out = []
        for i in range(0,N*num_pairs,num_pairs):
            curr_batch = combine_tensor[i:i+num_pairs,...]
            curr_idx_record = reshaped_pair_record[i:i+num_pairs,...]

            batch_embed = torch.empty(0,L+1,768).to(self.device)
            for i in range(ch):
                selected_tensor = curr_batch[curr_idx_record==i] # num x (L+1) x E
                #print('shape of selected_tensor: ',selected_tensor.shape)
                if selected_tensor.numel() != 0:
                    mean_tensor = selected_tensor.mean(dim=0) # L+1 x E
                    batch_embed = torch.cat((batch_embed,mean_tensor.unsqueeze(0)),dim=0) # num_ch x L+1 x E

            embed_out.append(batch_embed.unsqueeze(0)) # # N x num_ch x L+1 x 768
    
        embed_out = torch.cat(embed_out, dim=0).to(self.device) # N x num_ch x L+1 x 768

        # print('Embedding out shape',embed_out.shape)
        
        return embed_out
    

    def padding(self,in_cwt):
        N,_,L,F = in_cwt.shape
        # reshape to C,L,F
        max_length = self.max_length+1

        cwt_length = L
        ts_length = cwt_length+1
            
        padding_mask = torch.ones(N,self.max_length+1).to(self.device) # N, 388
        
        if ts_length > max_length:
            in_cwt = in_cwt[:, :,-max_length+1:, :] #3,max_length,65

        elif ts_length < max_length:
            in_cwt = torch.cat((torch.zeros(N,3,max_length-1-L, F).to(self.device), in_cwt), dim=2) # 3, max_len, 65
            padding_mask[:,:self.max_length-L+1] = 0

        return in_cwt, padding_mask



    def num_params(self):
        total_num = sum(p.numel() for p in self.parameters())
        train_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total of Parameters: {}M".format(round(total_num / 1e6, 2)))
        print("Train Parameters: {}M".format(round(train_num / 1e6, 2)))
        
    


if __name__ == '__main__':
    fake_x = torch.rand((3, 3, 128, 65))

    # model = PhysioModel(
    #     num_layers=6,
    #     upper_layer=2,
    #     emb_size=384,
    #     # data related
    #     in_channel=65
    # )
    # out = model(fake_x, input_mask=None, exchange_tks=None)
    # print("out shape:", out["embed"].shape) # (N, L+1, E = 3, 129, 384)
    # print("cls shape:", out["cls"].shape) # (N, E = 3, 384)
    # print("reconstruct shape:", out["rec"].shape) # (N, L+1, 1 = 3, 384, 1)