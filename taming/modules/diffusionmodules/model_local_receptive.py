import torch
import torch.nn as nn
from .model import Encoder, Normalize, AttnBlock


def attn_block_to_conv(attn_block):
    ''' Convert an attention block to a 1x1 convolution. '''
    channels = attn_block.in_channels
    return nn.Sequential(Normalize(channels), nn.Conv2d(channels, channels, 1), nn.Conv2d(channels, channels, 1))


def attn_block_to_masked_window_attn_block(attn_block, window_size):
    ''' Convert an attention block to a masked window attention block. '''
    channels = attn_block.in_channels
    return MaskedWindowAttnBlock(channels, window_size=window_size)


def create_similarity_mask(h, w, window_size=3):
    mask = torch.zeros((h*w, h*w), dtype=bool)
    for i in range(h):
        for j in range(w):
            start_row = max(0, i - window_size//2)
            end_row = min(h, i + window_size//2 + 1)
            start_col = max(0, j - window_size//2)
            end_col = min(w, j + window_size//2 + 1)

            start_index = i * w + j
            for ii in range(start_row, end_row):
                for jj in range(start_col, end_col):
                    index = ii * w + jj
                    mask[start_index, index] = True

    return mask


class MaskedWindowAttnBlock(AttnBlock):
    def __init__(self, *args, window_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        self.mask = None

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]

        # mask attention
        if self.mask is None:
            self.mask = create_similarity_mask(h, w, window_size=self.window_size).to(w_.device)
        w_ = w_.masked_fill(self.mask, float('-inf'))

        # finish computing attention
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_



class EncoderAtt2Conv(Encoder):
    
    ''' Swap the encoder's attention layers for 1x1 convolutions. '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # replace all the attention blocks with convolutions
        for i_level in range(self.num_resolutions):
            if len(self.down[i_level].attn)>0:
                self.down[i_level].attn = nn.ModuleList([attn_block_to_conv(attn_block) for attn_block in self.down[i_level].attn])
        self.mid.attn_1 = attn_block_to_conv(self.mid.attn_1)


class EncoderMaskedWindowAttn(Encoder):
    
    ''' Swap the encoder's attention layers for masked attention layers. '''

    def __init__(self, *args, window_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        for i_level in range(self.num_resolutions):
            if len(self.down[i_level].attn)>0:
                for i_attn in range(len(self.down[i_level].attn)):
                    self.down[i_level].attn[i_attn] = attn_block_to_masked_window_attn_block(self.down[i_level].attn[i_attn], window_size=window_size)
        self.mid.attn_1 = attn_block_to_masked_window_attn_block(self.mid.attn_1, window_size=window_size)
        