# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import einops
from einops.layers.torch import Rearrange
from typing import Union, Tuple, List
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops import rearrange
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from timm.models.vision_transformer import PatchEmbed
from timm.models.layers import trunc_normal_
from taming.modules.diffusionmodules.model import nonlinearity, ResnetBlock, AttnBlock, Normalize, Upsample


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         # we use xavier_uniform following official JAX ViT:
#         torch.nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.LayerNorm):
#         nn.init.constant_(m.bias, 0)
#         nn.init.constant_(m.weight, 1.0)
#     elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#         w = m.weight.data
#         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

# def get_2d_sincos_pos_embed(embed_dim, grid_size):
#     """
#     grid_size: int or (int, int) of the grid height and width
#     return:
#     pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
#     """
#     grid_size = (grid_size, grid_size) if type(grid_size) != tuple else grid_size
#     grid_h = np.arange(grid_size[0], dtype=np.float32)
#     grid_w = np.arange(grid_size[1], dtype=np.float32)
#     grid = np.meshgrid(grid_w, grid_h)  # here w goes first
#     grid = np.stack(grid, axis=0)

#     grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
#     pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

#     return pos_embed


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


# class Transformer(nn.Module):
#     def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int) -> None:
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for idx in range(depth):
#             layer = nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
#                                    PreNorm(dim, FeedForward(dim, mlp_dim))])
#             self.layers.append(layer)
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x

#         return self.norm(x)
    
class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor
    ):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

class EncoderVIT(nn.Module):
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
                 z_channels: int,  depth: int, heads: int, mlp_dim: int, channels: int = 3, dim_head: int = 64, **ignore_kwargs) -> None:
        super().__init__()
        
        # dim = z_channels
        image_height, image_width = image_size if isinstance(image_size, tuple) \
                                    else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \
                                    else (patch_size, patch_size)
        self.image_height = image_height
        self.image_width = image_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # en_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))

        # self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        # self.patch_dim = channels * patch_height * patch_width
        self.grid_size = image_size // patch_size
        # self.token_size = 12
        self.token_size = 16
        # self.num_latent_tokens = 128
        self.num_latent_tokens = 128

        # Transformer architecture
        self.width = 512
        self.num_layers = 8
        self.num_heads = 8

        # rearrange operation is converting the patches to 1D patch embeddings
        # positional embeddings are also 1D
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, self.width, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        # Example - img : 224 * 224, patch = 16 * 16
        # if i/p = (1, 3, 224, 224) --> Conv2d o/p = (1, self.width, 14, 14) --> O/p = (1, 196, self.width)

        # setting scale = 1/sq.rt(self.width). Normalizes wts, to prevent large initial values from dominating early training
        scale = self.width ** -0.5

        # learnable class token, that aggregates information from all patches
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))

        # poisitonal embeddings for the grid generated after patchifying
        # self.en_pos_embedding = nn.Parameter(torch.from_numpy(en_pos_embedding).float().unsqueeze(0))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        
        # positional embeddings for latent token 
        self.latent_token_positional_embedding = nn.Parameter(scale * torch.randn(self.num_latent_tokens, self.width))

        # normalizing input before transformers
        self.ln_pre = nn.LayerNorm(self.width)
        # transformer blocks
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        # normalize after all the transformer layers
        self.ln_post = nn.LayerNorm(self.width)
        # 1x1 conv, transformer embeddings --> discrete tokens. Final o/p: Transformed img in low dim space
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)

        # self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        # self.apply(init_weights)

    def resize_pos_embedding(self, new_shape):
        orig_posemb_h = self.image_height // self.patch_height
        orig_posemb_w = self.image_width // self.patch_width
        posemb = self.en_pos_embedding
        if orig_posemb_h == new_shape[0] and orig_posemb_w == new_shape[1]:
            return posemb
        posemb = rearrange(posemb, '1 (h w) d -> 1 d h w', h=orig_posemb_h, w=orig_posemb_w)
        posemb = torch.nn.functional.interpolate(posemb, new_shape, mode='bicubic')
        return rearrange(posemb, '1 d h w -> 1 (h w) d')

    def forward(self, img: torch.FloatTensor, latent_tokens) -> torch.FloatTensor:
        batch_size = img.shape[0]
        x = img
        x = self.to_patch_embedding(x) # shape : [batch_size, num_tokens, self.width], Eg: [1, 256, self.width]

        # adding class and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1) # shape: [batch_size, num_tokens+1, self.width]
        # adding positional embedding
        x = x + self.positional_embedding.to(x.dtype) # shape : [batch_size, grid_size ** 2 + 1, self.width]

        # expand so that each element gets latent tokens
        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        # concat positional embeddings to each token
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        
        # concat everything now : i.e x(patches and class token embeddings with pos embeddings) and latent tokens with positional embeddings
        x = torch.cat([x, latent_tokens], dim=1)

        # Transformer architecture
        # layer norm before transformers
        x = self.ln_pre(x)
        # permuting to [num_tokens, batch_size, self.width]
        x = x.permute(1, 0, 2)
        # passing through transformers
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2) # shape : [batch_size, num_tokens, self.width]

        # selecting all the tokens after class token and patch tokens, i.e selecting only latent tokens
        latent_tokens = x[:, 1+self.grid_size**2:]
        # normalizing after transformer block
        latent_tokens = self.ln_post(latent_tokens)

        latent_tokens = latent_tokens.reshape(batch_size, self.width, self.num_latent_tokens, 1)

        # applying 1x1 convolution
        latent_tokens = self.conv_out(latent_tokens)
        # reshaping the latent token sequence
        latent_tokens = latent_tokens.reshape(batch_size, self.token_size, 1, self.num_latent_tokens)

        return latent_tokens # final o/p shape : [batch_size, self.token_size, 1, self.num_latent_tokens]

    
#class EncoderVITPretrained(nn.Module):

class DecoderVIT(nn.Module):
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
                 z_channels: int, depth: int, heads: int, mlp_dim: int, dim_head: int=64, **ignore_kwargs) -> None:
        super().__init__()

        dim = z_channels
        image_height, image_width = image_size if isinstance(image_size, tuple) \
                                    else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \
                                    else (patch_size, patch_size)
        self.image_height = image_height
        self.image_width = image_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        #de_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.patch_embed = PatchEmbed(image_size, patch_size, dim, dim)
        #de_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))
        #self.de_pos_embedding = nn.Parameter(torch.from_numpy(de_pos_embedding).float().unsqueeze(0), requires_grad=False)
        scale = dim ** -0.5
        self.de_pos_embedding = nn.Parameter(scale*torch.randn(self.num_patches, dim))


        #self.to_pixels = nn.Linear(dim, patch_size**2 * 3)
        #self.to_pixels = nn.Sequential(nn.Linear(dim, patch_size**2 * 3),
        #                                nn.ConvTranspose2d(768, 3, 16, 16))

        
        self.norm = nn.LayerNorm(dim)
        self.MLP = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            #nn.Linear(dim * 4, dim)
        )

        self.to_pixel = nn.Sequential(
            nn.Linear(dim, 3*patch_size**2),
            Rearrange('b (hh ww) (c sh sw) -> b c (hh sh) (ww sw)', 
                      hh=image_height // patch_height, 
                      ww=image_height // patch_height, 
                      sh=patch_size, sw=patch_size))
            # rearrange(
            #     x, "b (hh ww) (c sh sw) -> b c (hh sh) (ww sw)",
            #     hh = image_height // patch_height, ww=image_height // patch_height,
            #     sh=patch_size, sw=patch_size
            # )
            #)



        # self.to_pixel = nn.Sequential(
        #     Rearrange('b (h w) c -> b c h w', h=image_height // patch_height),
        #     nn.ConvTranspose2d(dim, 3, kernel_size=patch_size, stride=patch_size)
        # )
        #                  #nn.ReLU(), 
                         #nn.ConvTranspose2d(256, 3, 4, 4))
        
        #self.to_pixels = nn.Sequential(
        #    nn.Linear(dim, 256),  # Example size, adjust as needed
        #    nn.ReLU(),
        #    nn.Linear(256, patch_size**2 * 3)  # Adjust output size based on image size and channels
        #)
        #self.apply(init_weights)
        self.init_parameters()
        
        #self.linear = nn.Linear(dim, z_channels)

    def init_parameters(self):
        if self.de_pos_embedding is not None:
            nn.init.normal_(self.de_pos_embedding, std=0.02)
        for block in self.transformer.layers:
            for module in block:
                if hasattr(module, 'weight') and module.weight is not None:
                    if len(module.weight.shape) > 1:
                        #nn.init.xavier_uniform_(module.weight)
                        trunc_normal_(module.weight, std=.02)    
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        trunc_normal_(self.MLP[0].weight, std=.02)
        trunc_normal_(self.to_pixel[0].weight, std=.02)


    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_height
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self, z: torch.FloatTensor) -> torch.FloatTensor:
        
        x = z.permute(0, 2, 3, 1).contiguous() # (B, H, W, D)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x + self.de_pos_embedding
        x = self.norm(x)
        x = self.transformer(x) # output (B, N, D)
        #x = x.permute(0,2,1).contiguous()
        #x = x.view(-1, x.size(1), self.patch_height, self.patch_width) 

        #x = self.norm(x)
        x = self.MLP(x)
        x = self.to_pixel(x) # from SimMIM model

        #x = self.unpatchify(x)
        #x = x.view(x.shape[0], x.shape[1], -1, self.patch_height, self.patch_width)
        #x = x.view(x.shape[0], -1, self.patch_height*(self.image_height//self.patch_height), self.patch_width*(self.image_width//self.patch_width))
        return x


class DecoderDINOCNN(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, num_layers=1, emb_dim=768, num_heads=12, num_mlp_ratio=4, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.emb_dim = emb_dim

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        
        self.decoder_dino_conv = nn.Conv2d(block_in, 768, kernel_size=1, stride=1, padding=0) # remove hard-coded emb_dim=768
        #self.decoder_dino_conv = nn.Conv2d(block_in, 1024, kernel_size=1, stride=1, padding=0) # remove hard-coded emb_dim=768

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        
        encoder_layer = TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=emb_dim*num_mlp_ratio)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # decoder dino output
        #decoder_dino_output = self.decoder_dino_conv(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        x = h
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1, self.emb_dim)
        x = self.transformer_encoder(x)
        x = x.reshape(x.shape[0], self.last_z_shape[2], self.last_z_shape[3], self.emb_dim)
        x = x.permute(0, 3, 1, 2)

        return h, x
    
# # Decoder with two branches: VIT for dino loss and CNN for reconstruction
# class DecoderDINOCNN(Decoder):
#     def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
#                  attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
#                  resolution, z_channels, encoder_model="VIT", num_layers=1, emb_dim=768, num_heads=12, num_mlp_ratio=4,  give_pre_end=False, **ignorekwargs):
#         super().__init__(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
#                          attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=resamp_with_conv,
#                          in_channels=in_channels, resolution=resolution, z_channels=z_channels, give_pre_end=give_pre_end)

#         #self.num_layers = num_layers
#         #self.num_heads = num_heads
#         #self.num_mlp_ratio = num_mlp_ratio
#         self.emb_dim = emb_dim
#         self.encoder_model = encoder_model

#         encoder_layer = TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=emb_dim*num_mlp_ratio)
#         self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.decoder_dino_conv = nn.Conv2d(z_channels, emb_dim, kernel_size=1, stride=1, padding=0) # remove hard-coded emb_dim=768
#         #self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
#         #self.decoder_dino_conv = nn.Conv2d(block_in, 768, kernel_size=1, stride=1, padding=0) # remove hard-coded emb_dim=768

#     def forward(self, z, dino_dist=True):

#         h = super().forward(z)
#         import pdb; pdb.set_trace()
#         # decoder dino output
#         if dino_dist:
#             #import pdb; pdb.set_trace()
#             if self.encoder_model == "CNN":
#                 #import pdb; pdb.set_trace()
#                 x = self.decoder_dino_conv(h) # 256, 768
#             else:
#                 x = h
#             x = x.permute(0, 2, 3, 1)
#             x = x.reshape(x.shape[0], -1, self.emb_dim)
#             x = self.transformer_encoder(x)
#             x = x.reshape(x.shape[0], self.last_z_shape[2], self.last_z_shape[3], self.emb_dim)
#             x = x.permute(0, 3, 1, 2)
#             return h, x

#         return h
