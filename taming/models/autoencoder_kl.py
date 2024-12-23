import timm
import torch
import torch.nn.functional as F
from einops import rearrange
from timm.layers.pos_embed import resample_abs_pos_embed
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR


from visual_tokenization.taming.models.vqgan_dino import ResNet50Block4Features
from visual_tokenization.taming.models.vqgan import VQModel2
from visual_tokenization.taming.util import instantiate_from_config




class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        noise = self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        x = self.mean + noise
        return x # , self.mean, self.logvar, self.std

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3]) , self.mean, self.logvar, self.std
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3]), self.mean, self.logvar, self.std

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
    





class AutoencoderKL(pl.LightningModule):
    def __init__(self, 
                encoder_config,
                decoder_config,
                quantizer_config,
                loss_config,
                grad_acc_steps=1,
                post_quantization_epoch=50,
                ckpt_path=None,
                ignore_keys=[],
                image_key="image",
                colorize_nlabels=None,
                monitor=None,
                distill_model_type = 'VIT_DINOv2', 
                ):
        super().__init__()
        
        self.automatic_optimization = False

        if not hasattr(decoder_config, 'params'):
            decoder_config.params = encoder_config.params
        #print (encoder_config)
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.loss = instantiate_from_config(loss_config)
        self.quantize = instantiate_from_config(quantizer_config)
        
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.distill_model_type = distill_model_type
        self.grad_acc_steps = grad_acc_steps
        self.post_quantization_epoch = post_quantization_epoch
        
        self.patch_size = encoder_config.params['patch_size']
        self.image_size = encoder_config.params['image_size']
        self.encoder_normalize_embedding = encoder_config.params.get("normalize_embedding", False)
        

        if distill_model_type:
            self.distill, self.post_quant_conv_distill = self.init_distill_model(distill_model_type)
        
        self.quant_conv = torch.nn.Conv2d(encoder_config.params["z_channels"], encoder_config.params['e_dim'], 1)
        self.post_quant_conv = torch.nn.Conv2d(decoder_config.params['e_dim'], decoder_config.params["z_channels"], 1)
        
        self.encoder_normalize_embedding = encoder_config.params.get("normalize_embedding", False)
        self.grad_acc_steps = grad_acc_steps

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def on_load_checkpoint(self, checkpoint):
        self.load_state_dict(checkpoint['state_dict'], strict=False)
    def on_load_checkpoint(self, checkpoint):
        self.load_state_dict(checkpoint['state_dict'], strict=False)
        # Restore optimizer and scheduler states if necessary
        if 'optimizer_states' in checkpoint:
            self.optimizer_states = checkpoint['optimizer_states']
        if 'lr_schedulers' in checkpoint:
            self.lr_schedulers = checkpoint['lr_schedulers']

    
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    


    def get_input(self, batch, k):
        if isinstance(batch, dict):
            x = batch[k]
        else:
            x = batch
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        
        return x.float()
    
    def init_distill_model(self, distill_model_type):
        if distill_model_type == 'VIT_DINO':
            distill = timm.create_model('timm/vit_base_patch16_224.dino', img_size=self.image_size, pretrained=True).eval() # DINO VIT-Base emb_dim:768
            post_quant_conv_distill = torch.nn.Conv2d(self.decoder_config.params['e_dim'], self.decoder_config.params['z_channels'], 1)
        elif distill_model_type == 'VIT_DINOv2':
            distill = timm.create_model('timm/vit_base_patch14_dinov2.lvd142m', img_size=224, pretrained=True).eval() #.cuda().eval() # DINO VIT-Base emb_dim:768
            post_quant_conv_distill = torch.nn.Conv2d(self.decoder_config.params['e_dim'], self.decoder_config.params['z_channels'], 1)
        elif distill_model_type == 'VIT_DINOv2_large':
            distill = timm.create_model('timm/vit_large_patch14_dinov2.lvd142m', img_size=224, pretrained=True).eval() #.cuda().eval() # DINO VIT-Base emb_dim:768
            post_quant_conv_distill = torch.nn.Conv2d(self.decoder_config.params['e_dim'], self.decoder_config.params['z_channels'], 1)
        elif distill_model_type == 'VIT_DINOv2_large_reg4':
            distill = timm.create_model('timm/vit_large_patch14_reg4_dinov2.lvd142m', img_size=224, pretrained=True).eval() #.cuda().eval() # DINO VIT-Base emb_dim:768    
            post_quant_conv_distill = torch.nn.Conv2d(self.decoder_config.params['e_dim'], self.decoder_config.params['z_channels'], 1)
        elif distill_model_type == 'SAM_VIT':
            distill = timm.create_model('samvit_large_patch16.sa1b', pretrained=True)
            post_quant_conv_distill = torch.nn.Identity()
        elif distill_model_type == 'SAM_VIT_w_conv':
            distill = timm.create_model('samvit_large_patch16.sa1b', pretrained=True)
            post_quant_conv_distill = torch.nn.Conv2d(self.decoder_config.params['e_dim'], self.decoder_config.params['z_channels'], 1) # z_channels = 256
        elif distill_model_type == 'CNN_DINO':
            dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            distill = ResNet50Block4Features(dino_model).cuda().eval()
            post_quant_conv_distill = torch.nn.Sequential(
                torch.nn.Conv2d(self.decoder_config.params['e_dim'], self.decoder_config.params['z_channels'], kernel_size=1, stride=1, padding=0),
                torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            )
        elif distill_model_type == 'depth_anything_VIT_L14':
            distill = timm.create_model('vit_large_patch14_dinov2.lvd142m', img_size=224, pretrained=False)
            pretrained_state_dict = torch.load("./pretrained_models/depth_anything_vitl14.pth")
            prefix = 'pretrained.'
            new_state_dict = {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in pretrained_state_dict.items()}
            new_state_dict['pos_embed'] = resample_abs_pos_embed(new_state_dict['pos_embed'], new_size=(16, 16)) # H x W
            distill.load_state_dict(new_state_dict, strict=False)
            post_quant_conv_distill = torch.nn.Conv2d(self.decoder_config.params['e_dim'], self.decoder_config.params['z_channels'], 1)
        return distill, post_quant_conv_distill


    def encode(self, x, quantize=True):
        self.encoder.train() if not quantize and self.training else self.encoder.eval()
        with torch.no_grad() if quantize else torch.enable_grad():
            h = self.encoder(x)
            moments = self.quant_conv(h)
            # if self.encoder_normalize_embedding:
            #     moments = F.normalize(moments, p=2, dim=1)
        posterior = DiagonalGaussianDistribution(moments)
        emb_loss = torch.tensor(0.0).to(h.device)
        quant = None
        info = None
        if quantize:
            z = posterior.mode()
            quant, emb_loss, info = self.quantize(z)
            return  quant, emb_loss, info
        else:
            emb_loss = torch.tensor(0.0).to(h.device)
            return posterior, emb_loss, None

    def decode(self, z):
        z_post_quant = self.post_quant_conv(z)
        dec = self.decoder(z_post_quant)
        return dec, self.post_quant_conv_distill(z)
    
    def decode_code(self, code_b): # input shape: b d h w --> output shape: b c h w
        quant_b = self.quantize.get_codebook_entry(rearrange(code_b[:,0,:,:], 'b h w -> b (h w)'), 0)
        for i in range(1, code_b.size(1)):
            quant_b += self.quantize.get_codebook_entry(rearrange(code_b[:,i,:,:], 'b h w -> b (h w)'), i)
        quant_b = rearrange(quant_b, 'b (h w) z -> b z h w', h=code_b.size(2))  
        dec = self.decode(quant_b, None)
        return dec

    def forward(self, input, sample_posterior=True, quantize=True):        
        if quantize:
            quant, emb_loss, info = self.encode(input, quantize)
            dec, decoder_distill_output = self.decode(quant)
            return dec, quant, emb_loss, decoder_distill_output
        else:
            posterior, emb_loss, info = self.encode(input, quantize)
            if sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()
            dec, decoder_distill_output = self.decode(z)
            return dec, posterior, emb_loss, decoder_distill_output
    

    def distill_loss(self, distill_output, decoder_distill_output):
        if 'VIT' in self.distill_model_type:
            if 'reg4' in self.distill_model_type:
                distill_output = distill_output[:, 5:, :] # [CLS, Register*4, Embeddings]
            elif 'reg4' not in self.distill_model_type and 'DINO' in self.distill_model_type:
                distill_output = distill_output[:, 1:, :] # uncomment for DINOv1 
            elif 'depth_anything' in self.distill_model_type:
                distill_output = distill_output[:, 1:, :]
            elif self.distill_model_type == 'SAM_VIT':
                distill_output = distill_output.permute(0, 2, 3, 1).contiguous().view(distill_output.shape[0], -1, distill_output.shape[1])
                distill_output = F.normalize(distill_output, p=2, dim=2) # without post_conv layer
            elif self.distill_model_type == 'SAM_VIT_w_conv':
                distill_output = distill_output.permute(0, 2, 3, 1).contiguous().view(distill_output.shape[0], -1, distill_output.shape[1])
                # without L2 normalization
            distill_output = distill_output.permute(0, 2, 1).contiguous()
        
        elif self.distill_model_type == 'CNN':
            distill_output = distill_output.view(distill_output.shape[0], distill_output.shape[1], -1)
        decoder_distill_output = decoder_distill_output.view(decoder_distill_output.shape[0], decoder_distill_output.shape[1], -1)
        #print (f'distill_output.shape: {distill_output.shape} decoder_distill_output.shape: {decoder_distill_output.shape}')
        cos_similarity = F.cosine_similarity(decoder_distill_output, distill_output, dim=1)
        cosine_loss = 1 - cos_similarity
        distill_loss = cosine_loss.mean()
        return distill_loss

    def training_step(self, batch, batch_idx):
        opt_ae, opt_disc = self.optimizers()
        [scheduler_ae_warmup, scheduler_disc_warmup] = self.lr_schedulers()

        x = self.get_input(batch, self.image_key)
        
        
        if self.current_epoch >= self.post_quantization_epoch:
            xrec, quant, emb_loss, decoder_distill_output = self(x, quantize=True)
            posterior = None
        else:
            xrec, posterior, emb_loss, decoder_distill_output = self(x, quantize=False)

        # Calculate distillation loss
        with torch.no_grad():
            if 'VIT' in self.distill_model_type :
                if 'VIT_DINOv2' in self.distill_model_type or 'depth_anything' in self.distill_model_type:
                    x_224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                    distill_output = self.distill.forward_features(x_224)
                else: # for VIT-DINOv1, VIT-SAM models
                    distill_output = self.distill.forward_features(x)
            elif self.distill_model_type == 'CNN':
                distill_output = self.distill(x)
        
        
        if self.current_epoch < self.post_quantization_epoch and self.loss.distill_loss_weight > 0:
            distill_loss = self.distill_loss(distill_output, decoder_distill_output)
        else:
            distill_loss = torch.tensor(0.0).to(distill_output.device)



        optimizer_idx = 0
        if self.current_epoch >= self.post_quantization_epoch:
            aeloss, log_dict_ae = self.loss(quant, distill_loss, emb_loss, x, xrec, optimizer_idx, self.global_step, quantize=True,
                                        last_layer=self.get_last_layer(), split="train")
        else:
            aeloss, log_dict_ae = self.loss(posterior, distill_loss, emb_loss, x, xrec, optimizer_idx, self.global_step, quantize=False,
                                            last_layer=self.get_last_layer(), split="train")
        self.log("train/distill_loss", distill_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()

        
        optimizer_idx = 1
        # train the discriminator
        if self.current_epoch >= self.post_quantization_epoch:
            discloss, log_dict_disc  = self.loss(quant, distill_loss, emb_loss, x, xrec, optimizer_idx, self.global_step, quantize=True,
                                        last_layer=self.get_last_layer(), split="train")
        else:
            discloss, log_dict_disc  = self.loss(posterior, distill_loss, emb_loss, x, xrec, optimizer_idx, self.global_step, quantize=False,
                                            last_layer=self.get_last_layer(), split="train")

        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        discloss = discloss / self.grad_acc_steps
        self.manual_backward(discloss)
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_disc.step()
            opt_disc.zero_grad()
            scheduler_disc_warmup.step()



    def validation_step(self, batch, batch_idx):
        
        x = self.get_input(batch, self.image_key)
        
        
        if self.current_epoch >= self.post_quantization_epoch:
            xrec, quant, emb_loss, decoder_distill_output = self(x, quantize=True)
            posterior = None
        else:
            xrec, posterior, emb_loss, decoder_distill_output = self(x, quantize=False)

        # Calculate distillation loss
        with torch.no_grad():
            if 'VIT' in self.distill_model_type :
                if 'VIT_DINOv2' in self.distill_model_type or 'depth_anything' in self.distill_model_type:
                    x_224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                    distill_output = self.distill.forward_features(x_224)
                else: # for VIT-DINOv1, VIT-SAM models
                    distill_output = self.distill.forward_features(x)
            elif self.distill_model_type == 'CNN':
                distill_output = self.distill(x)

        if self.current_epoch < self.post_quantization_epoch and self.loss.distill_loss_weight > 0:
            distill_loss = self.distill_loss(distill_output, decoder_distill_output)
        else:
            distill_loss = torch.tensor(0.0).to(distill_output.device)


        optimizer_idx = 1
        if self.current_epoch >= self.post_quantization_epoch:
            discloss, log_dict_disc  = self.loss(quant, distill_loss, emb_loss, x, xrec, optimizer_idx, self.global_step, quantize=True,
                                        last_layer=self.get_last_layer(), split="val")
        else:
            discloss, log_dict_disc  = self.loss(posterior, distill_loss, emb_loss, x, xrec, optimizer_idx, self.global_step, quantize=False,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        optimizer_idx = 0
        if self.current_epoch >= self.post_quantization_epoch:
            aeloss, log_dict_ae = self.loss(quant, distill_loss, emb_loss, x, xrec, optimizer_idx, self.global_step, quantize=True,
                                        last_layer=self.get_last_layer(), split="val")
        else:
            aeloss, log_dict_ae = self.loss(posterior, distill_loss, emb_loss, x, xrec, optimizer_idx, self.global_step, quantize=False,
                                            last_layer=self.get_last_layer(), split="val")
        self.log("val/distill_loss", distill_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return self.log_dict


    def get_warmup_scheduler(self, optimizer, warmup_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                #print(f"Step: {step}, Warmup: {warmup_steps}, Warmup Start LR: {warmup_start_lr}, Max LR: {max_lr}")
                return step/warmup_steps
            # After warmup_steps, we just return 1. This could be modified to implement your own schedule
            return 1.0
        
        return LambdaLR(optimizer, lr_lambda)
    

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.post_quant_conv_distill.parameters()),
                                  lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        
        #warmup_start_lr = 1e-7
        #max_lr = self.learning_rate
        scheduler_ae_warmup = self.get_warmup_scheduler(opt_ae, self.loss.warmup_steps)
        scheduler_disc_warmup = self.get_warmup_scheduler(opt_disc, self.loss.warmup_steps)
        

        return [opt_ae, opt_disc], [scheduler_ae_warmup, scheduler_disc_warmup]
    

    
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec_cont, _, _, _ = self(x, quantize=False)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec_cont.shape[1] > 3
            x = self.to_rgb(x)
            xrec_cont = self.to_rgb(xrec_cont)
        log["inputs"] = x
        log["reconstructions_cont"] = xrec_cont

                
        if self.current_epoch >= self.post_quantization_epoch:
            xrec_quant, _, _, _ = self(x, quantize=True)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec_cont.shape[1] > 3
                xrec_quant = self.to_rgb(xrec_quant)
            log["reconstructions_quant"] = xrec_quant

        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


    def get_last_layer(self):
        return self.decoder.conv_out.weight
    

