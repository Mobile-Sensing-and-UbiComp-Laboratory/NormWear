import torch
import torch.nn as nn
from .pos_embed import tAPE

from timm.models.layers import to_2tuple

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







if __name__ == '__main__':
    # patch_emb = PatchEmbed_new(img_size=(387,65), patch_size=(9,5), in_chans=3, embed_dim=64, stride=(9,5))
    # input = torch.rand(8,3,387,65)
    # output = patch_emb(input)
    # print(output.shape) # (8,559,64)

    # patch_emb = PatchEmbed3D_new(video_size=(6,224,224), patch_size=(2,16,16), in_chans=3, embed_dim=768, stride=(2,16,16))
    # input = torch.rand(8,3,6,224,224)
    # output = patch_emb(input)
    #print(output.shape) # (8,64)

    patch_emb = PatchEmbed_ts(ts_len=387,patch_size=9,stride=9)
    input = torch.randn(6,387)
    output = patch_emb(input)
    print(output.shape)
    print(patch_emb.patch_size)