import random
import numpy as np
import torch
import timm
import torch.nn.functional as F
from torch import Tensor, nn

from einops.layers.torch import Rearrange

from typing import Any, Optional, Sequence, Tuple, Union


class Encoder(nn.Module):
    def __init__(
        self, 
        image_size: Union[Tuple[int, int], int], 
        channels: int = 3, 
        pretrained_encoder = 'MAE',
        
        patch_size: int = 16,
        z_channels: int = 768,
        normalize_embedding: bool = True,
        **ignore_kwargs
    ) -> None:
        # Initialize parent class with the first patch size
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.normalize_embedding = normalize_embedding
        self.z_channels = z_channels
        

        self.init_transformer(pretrained_encoder)


    def init_transformer(self, pretrained_encoder):
        if pretrained_encoder == 'VIT_DINO':
            pretrained_encoder_model = 'timm/vit_base_patch16_224.dino'
        elif pretrained_encoder == 'VIT_DINOv2':
            pretrained_encoder_model = 'timm/vit_base_patch14_dinov2.lvd142m'
        elif pretrained_encoder == 'MAE':
            pretrained_encoder_model = 'timm/vit_base_patch16_224.mae'
        elif pretrained_encoder == 'MAE_VIT_L':
            pretrained_encoder_model = 'timm/vit_large_patch16_224.mae'
        elif pretrained_encoder == 'VIT':
            pretrained_encoder_model = 'timm/vit_large_patch32_224.orig_in21k'
        elif pretrained_encoder == 'CLIP32':
            pretrained_encoder_model = 'timm/vit_base_patch32_clip_224.openai'
        elif pretrained_encoder == 'CLIP':
            pretrained_encoder_model = 'timm/vit_base_patch16_clip_224.openai'
       

       # TODO: why do we initialize the encoder with the pretrained model and then overwrite the weights with the same model?
        self.encoder = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=False, dynamic_img_size=True).train()
        pretrained_model = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=True)
        """Initialize weights of target_model with weights from source_model."""
        with torch.no_grad():
            for target_param, source_param in zip(self.encoder.parameters(), pretrained_model.parameters()):
                target_param.data.copy_(source_param.data)
    
    
    
    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        h = self.encoder.forward_features(img)[:,1:].permute(0, 2, 1).contiguous()
        num_patches = self.image_size // self.patch_size
        h = h.reshape(h.shape[0], -1, h.shape[2]//num_patches, h.shape[2]//num_patches)
        return h
