import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from taming.util import instantiate_from_config
from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.diffusionmodules.model_vit import EncoderVIT, DecoderDINOCNN
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
# from taming.modules.vqvae.quantize import VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer
from taming.models.vqgan import VQModel, VQModel2
from taming.models.vqgan_with_entropy_loss import VQModel2WithEntropyLoss
from taming.modules.vqvae.quantize_with_entropy_loss import VectorQuantizer2WithEntropyLoss

import timm
from einops import rearrange
from torchvision import models
from timm.layers.pos_embed import resample_abs_pos_embed

class ResNet50Block4Features(models.ResNet):
    def __init__(self, pretrained_model):
        super(ResNet50Block4Features, self).__init__(block=models.resnet.Bottleneck, layers=[3, 4, 6, 3])
        #remove fc layer from Resnet50
        self.fc = torch.nn.Identity()
        self.load_state_dict(pretrained_model.state_dict())  # Copy weights from the pretrained model

    def forward(self, x):
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Block groups
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x) # 8x8x2048

        x = self.avgpool(features)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return features

class VQModel2WithEntropyDINOLoss(VQModel2WithEntropyLoss):
    def __init__(self, 
                 encoder_config,
                 decoder_config,
                 quantizer_config,
                 loss_config,
                 grad_acc_steps=1,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 entropy_loss_weight_scheduler_config=None,
                 dino_model_type = 'VIT_DINOv2', 
                 ):
        super().__init__(encoder_config, decoder_config, quantizer_config, loss_config, 
                         grad_acc_steps, ckpt_path, ignore_keys, image_key, colorize_nlabels, 
                         monitor, 
                         entropy_loss_weight_scheduler_config, )
        n_embed = quantizer_config.params['n_e']
        embed_dim = quantizer_config.params['e_dim']
        image_size = encoder_config.params['image_size']
        quantizer_normalize_embedding = quantizer_config.params.get("normalize_embedding", False)
        self.num_latent_tokens = 128
        # self.num_latent_tokens = 144
        self.loss = instantiate_from_config(loss_config)
        self.quantizer_normalize_embedding = quantizer_normalize_embedding
        self.quantize = instantiate_from_config(quantizer_config)
        self.entropy_loss_weight_scheduler = instantiate_from_config(entropy_loss_weight_scheduler_config)
        self.grad_acc_steps = grad_acc_steps
        self.width = 512
        self.scale = self.width ** -0.5
        self.latent_tokens = nn.Parameter(self.scale * torch.randn(self.num_latent_tokens, self.encoder.width))
       
        # self.dino_model_type = dino_model_type
        # if dino_model_type == 'VIT_DINO':
        #     self.dino = timm.create_model('timm/vit_base_patch16_224.dino', img_size=image_size, pretrained=True).eval() # DINO VIT-Base emb_dim:768
        #     self.post_quant_conv_dino = torch.nn.Conv2d(quantizer_config.params['e_dim'], decoder_config.params['z_channels'], 1)
        # elif dino_model_type == 'VIT_DINOv2':
        #     self.dino = timm.create_model('timm/vit_base_patch14_dinov2.lvd142m', img_size=224, pretrained=True).eval()  # DINO VIT-Base emb_dim:768
        #     self.post_quant_conv_dino = torch.nn.Conv2d(quantizer_config.params['e_dim'], decoder_config.params['z_channels'], 1)

        self.encoder_normalize_embedding = encoder_config.params.get("normalize_embedding", False)

    def entropy_loss_weight_scheduling(self):
        self.loss.entropy_loss_weight = self.entropy_loss_weight_scheduler(self.global_step)

    def encode(self, x):
        h = self.encoder(x, latent_tokens = self.latent_tokens)
        # h = self.quant_conv(h)
        if self.encoder_normalize_embedding:
            h = F.normalize(h, p=2, dim=1)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        # dino_conv_out = self.post_quant_conv_dino(quant)
        # quant2 = self.post_quant_conv(quant)
        # print("-----Decoder Input-----: ", quant.shape)
        return self.decoder(quant)

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        # print('-----Visualization-----: ')
        # visualize = self.visualize_tokens(quant)
        dec = self.decode(quant)
        # print("-----Decoder Output-----: ", dec.shape)
        return dec, diff, _

    def dino_loss(self, dino_output, decoder_dino_output):
        dino_output = dino_output[:, 1:, :]
        dino_output = dino_output.permute(0, 2, 1).contiguous()
        decoder_dino_output = decoder_dino_output.view(decoder_dino_output.shape[0], decoder_dino_output.shape[1], -1)
        cos_similarity = F.cosine_similarity(decoder_dino_output, dino_output, dim=1)
        cosine_loss = 1 - cos_similarity
        dino_loss = cosine_loss.mean()
        return dino_loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  # list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  # list(self.post_quant_conv_dino.parameters()),
                                  lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))

        scheduler_ae_warmup = self.get_warmup_scheduler(opt_ae, self.loss.warmup_steps, self.loss.min_lr_multiplier)
        scheduler_disc_warmup = self.get_warmup_scheduler(opt_disc, self.loss.warmup_steps, self.loss.min_lr_multiplier)
        
        return [opt_ae, opt_disc], [scheduler_ae_warmup, scheduler_disc_warmup]

    def training_step(self, batch, batch_idx):
        self.entropy_loss_weight_scheduling()
        self.log("train/enropy_loss_weight", self.loss.entropy_loss_weight, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        opt_ae, opt_disc = self.optimizers()
        [scheduler_ae_warmup, scheduler_disc_warmup] = self.lr_schedulers()

        x = self.get_input(batch, self.image_key)
        xrec, qloss, decoder_dino_output = self(x)
        # with torch.no_grad():
        #     if 'VIT_DINOv2' in self.dino_model_type:
        #         x_224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        #         dino_output = self.dino.forward_features(x_224)
        #     else: # for VIT-DINOv1, VIT-SAM models
        #         dino_output = self.dino.forward_features(x)

        # dino_loss = self.dino_loss(dino_output, decoder_dino_output)

        optimizer_idx = 1
        discloss, log_dict_disc = self.loss(codebook_entropy_losses=qloss, dino_loss=None, inputs=x, reconstructions=xrec, optimizer_idx=1, global_step=self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        discloss = discloss / self.grad_acc_steps
        self.manual_backward(discloss)
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_disc.step()
            opt_disc.zero_grad()
            scheduler_disc_warmup.step()

        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(codebook_entropy_losses=qloss, dino_loss=None, inputs=x, reconstructions=xrec, optimizer_idx=0, global_step=self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        # self.log("train/dino_loss", dino_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()
    
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, decoder_dino_output = self(x)
        # with torch.no_grad():
        #     # resize image x to 224x224
        #     if 'VIT_DINOv2' in self.dino_model_type or 'depth_anything' in self.dino_model_type:
        #         x_224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        #         dino_output = self.dino.forward_features(x_224)
        #     else: # for VIT-DINOv1, VIT-SAM models
        #         dino_output = self.dino.forward_features(x)

        # dino_loss = self.dino_loss(dino_output, decoder_dino_output)
        aeloss, log_dict_ae = self.loss(codebook_entropy_losses=qloss, dino_loss=None, inputs=x, reconstructions=xrec, optimizer_idx=0, global_step=self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        # self.log("val/dino_loss", dino_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        discloss, log_dict_disc = self.loss(codebook_entropy_losses=qloss, dino_loss=None, inputs=x, reconstructions=xrec, optimizer_idx=1, global_step=self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_disc)
        return self.log_dict

class VQModel2WithEntropyDINOLossMAEinit(VQModel2WithEntropyDINOLoss):
    def __init__(self, 
                 encoder_config,
                 decoder_config,
                 quantizer_config,
                 loss_config,
                 grad_acc_steps=1,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 entropy_loss_weight_scheduler_config=None,
                #  dino_model_type = 'VIT_DINOv2', # 'VIT_DINO' or 'CNN' or VIT_DINOv2, VIT_DINOv2_large_reg4, SAM_VIT
                #  pretrained_encoder = 'MAE', #'VIT_DINOv2',
                 ):
        super().__init__(encoder_config, decoder_config, quantizer_config, loss_config, 
                         grad_acc_steps, ckpt_path, ignore_keys, image_key, colorize_nlabels, 
                         monitor, 
                         entropy_loss_weight_scheduler_config 
                        #  dino_model_type
                         )
        self.patch_size = encoder_config.params['patch_size']
        self.image_size = encoder_config.params['image_size']
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_latent_tokens = 128
        # self.num_latent_tokens = 144
        self.encoder.width = 512
        self.width = 512
        self.scale = scale = self.width ** -0.5
        self.latent_tokens = nn.Parameter(self.scale * torch.randn(self.num_latent_tokens, self.encoder.width))


        # if pretrained_encoder:
        #     if pretrained_encoder == 'VIT_DINO':
        #         pretrained_encoder_model = 'timm/vit_base_patch16_224.dino'
        #     elif pretrained_encoder == 'VIT_DINOv2':
        #         pretrained_encoder_model = 'timm/vit_base_patch14_dinov2.lvd142m'
        #     elif pretrained_encoder == 'MAE':
        #         pretrained_encoder_model = 'timm/vit_base_patch16_224.mae'
            
        #     self.encoder = timm.create_model(pretrained_encoder_model, img_size=self.image_size, pretrained=False).train()
        #     pretrained_model = timm.create_model(pretrained_encoder_model, img_size=self.image_size, pretrained=True)
        #     self.initialize_weights(self.encoder, pretrained_model)
        self.encoder = instantiate_from_config(encoder_config)

    # def initialize_weights(self, target_model, source_model):
    #     """Initialize weights of target_model with weights from source_model."""
    #     with torch.no_grad():
    #         for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
    #             target_param.data.copy_(source_param.data)

    def encode(self, x):
        # h = self.encoder.forward_features(x)[:,1:].permute(0, 2, 1).contiguous()
        # print("-----Encoder Input-----: ", x.shape)
        h = self.encoder(x, latent_tokens=self.latent_tokens)
        # print("-----Encoder Output-----: ", h.shape)
        # num_patches = self.image_size // self.patch_size
        # h = h.reshape(h.shape[0], -1, h.shape[2]//num_patches, h.shape[2]//num_patches)
        # h = self.quant_conv(h)
        if self.encoder_normalize_embedding:
            h = F.normalize(h, p=2, dim=1)
        quant, emb_loss, info = self.quantize(h)
        # print("-----Quantizer Output-----: ", quant.shape)
        return quant, emb_loss, info

    # def visualize_tokens(self, quant):
    #     """ Visualize token representations from the encoder """
    #     self.eval()  # Set model to eval mode
    #     with torch.no_grad():
    #         # quant, _, _ = self.encode(x.unsqueeze(0))  # Encode image
    #         quant = quant.squeeze(0).cpu().numpy()  # Remove batch dim
    
    #     # Reshape tokens to approximate 2D structure
    #     num_tokens = self.num_latent_tokens  # Should be 128 in your case
    #     grid_size = int(num_tokens ** 0.5)  # Approximate square grid
    #     print("----------Visualization Quant Shape-----------: ", quant.shape)

    #     if quant.shape[-1] == num_tokens:
    #         token_grid = quant.reshape(grid_size, grid_size, -1).mean(axis=-1)  # Average over channels
    #     else:
    #         print(f"Unexpected token shape: {quant.shape}")
    #         return
    
    #     # Plot the tokenized representation
    #     fig = plt.figure(figsize=(6, 6))
    #     plt.imshow(token_grid, cmap="gray")
    #     plt.colorbar()
    #     plt.title("Tokenized Image Representation")
    #     plt.axis("off")
    #     plt.savefig("titok_project/token_compression/plots")
    #     # plt.show()
         

class VQModelDino(VQModel):
    def __init__(self,
                 ddconfig,
                 encoder_params,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 encoder_model='CNN',
                 decoder_model='CNN',
                 normalize_embedding=False,
                 dino_dist=False,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__(ddconfig, lossconfig, n_embed, embed_dim, ckpt_path, ignore_keys, image_key, colorize_nlabels, monitor, remap, sane_index_shape)   
        self.automatic_optimization = False
        self.image_key = image_key
        self.normalize_embedding = normalize_embedding
        self.dino_dist = dino_dist
        self.decoder_model = decoder_model
        self.encoder_model = encoder_model
        if self.encoder_model == 'CNN':
            self.encoder = Encoder(**ddconfig)
        elif self.encoder_model == 'VIT':
            self.encoder = EncoderVIT(**encoder_params)

        if self.decoder_model == 'CNN':
            raise NotImplementedError("Decoder with dino distillation not implemented.")
            self.decoder = Decoder(**ddconfig)

        self.dino = timm.create_model('timm/vit_base_patch16_224.dino', pretrained=True).cuda().eval() # DINO VIT-Base emb_dim:768
        #self.dino = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m', pretrained=True, num_classes=0,).cuda().eval() # DINOv2 VIT-large 14, emb_dim:1024, input size:518
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        if self.normalize_embedding:
            h = F.normalize(h, p=2, dim=1)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        if self.dino_dist:
            return self.decoder(quant, dino_dist=True)
        else:
            return self.decoder(quant)

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        if self.dino_dist:
            dec, dino_conv_out = self.decode(quant)
            return dec, diff, dino_conv_out
        else:
            dec = self.decode(quant)
            return dec, diff

    def dino_loss(self, dino_output, decoder_dino_output):
        dino_output = dino_output[:, 1:, :] # uncomment for DINOv1 
        dino_output = dino_output.permute(0, 2, 1).contiguous()
        decoder_dino_output = decoder_dino_output.view(decoder_dino_output.shape[0], decoder_dino_output.shape[1], -1)
        cos_similarity = F.cosine_similarity(decoder_dino_output, dino_output, dim=2)
        cosine_loss = 1 - cos_similarity
        dino_loss = cosine_loss.mean()
        return dino_loss

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        if self.dino_dist:
            xrec, qloss, decoder_dino_output = self(x)
            with torch.no_grad():
                dino_output = self.dino.forward_features(x)
                
            dino_loss = self.dino_loss(dino_output, decoder_dino_output)
        else:
            xrec, qloss = self(x)
            dino_loss = torch.tensor([0.0])
        opt_ae, opt_disc = self.optimizers()

        optimizer_idx = 1
        # discriminator
        discloss, log_dict_disc = self.loss(qloss, dino_loss, x, xrec, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

        optimizer_idx = 0
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, dino_loss, x, xrec, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        if self.dino_dist:
            self.log("train/dino_loss", dino_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        if self.dino_dist:
            xrec, qloss, decoder_dino_output = self(x)
            with torch.no_grad():
                #import pdb; pdb.set_trace()
                dino_output = self.dino.forward_features(x)
                dino_loss = self.dino_loss(dino_output, decoder_dino_output)
        else:
            xrec, qloss = self(x)
            dino_loss = torch.tensor([0.0])
        
        aeloss, log_dict_ae = self.loss(qloss, dino_loss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        if self.dino_dist:
            self.log("val/dino_loss", dino_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        discloss, log_dict_disc = self.loss(qloss, dino_loss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        #self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if self.dino_dist:
            xrec, _, _ = self(x)
        else:
            xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

