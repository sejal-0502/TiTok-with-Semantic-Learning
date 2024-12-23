import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, warmup_steps = 1000, distill_loss_weight=0.1, codebook_weight= 1.0, use_actnorm=False, disc_conditional=False, beta_1=0.5, beta_2=0.9,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.distill_loss_weight = distill_loss_weight
        self.warmup_steps = warmup_steps
        self.codebook_weight = codebook_weight
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, posteriors, distill_loss, codebook_loss, x, xrec,  optimizer_idx,
                global_step, quantize, last_layer=None, cond=None, split="train",
                weights=None):
        

        # now the GAN part
        if optimizer_idx == 0:
            rec_loss = torch.abs(x.contiguous() - xrec.contiguous())
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(x.contiguous(), xrec.contiguous())
                rec_loss = rec_loss + self.perceptual_weight * p_loss
            else:
                p_loss = torch.tensor([0.0])

            
            if quantize:
                nll_loss = rec_loss.mean()
                # weighted_nll_loss = nll_loss
                kl_loss = torch.tensor(0.0)
                mean = torch.tensor(0.0)
                logvar = torch.tensor(0.0)
                std = torch.tensor(0.0)
            else:
                nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
                # weighted_nll_loss = nll_loss
                # if weights is not None:
                #     weighted_nll_loss = weights*nll_loss
                # weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
                # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
                # kl_loss, mean, logvar, std  = posteriors.kl()
                # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                nll_loss = torch.mean(nll_loss)
                kl_loss, mean, logvar, std  = posteriors.kl()
                kl_loss = torch.mean(kl_loss) / (mean.shape[1] * mean.shape[2] * mean.shape[3])

            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(xrec.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((xrec.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss + self.distill_loss_weight * distill_loss.mean()
            loss += self.codebook_weight * codebook_loss.mean()
            log = {
                   "{}/kl_loss".format(split): kl_loss.detach().mean(),
                   "{}/codebook_loss".format(split): codebook_loss.clone().detach().mean(),
                #    "{}/weighted_nll_loss".format(split): weighted_nll_loss.clone().detach().mean(),
                    "{}/total_loss".format(split): loss.clone().detach().mean(),
                    "{}/logvar".format(split): self.logvar.detach(),
                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
                    "{}/nll_loss".format(split): nll_loss.detach(),
                    "{}/d_weight".format(split): d_weight.detach(),
                    "{}/p_loss".format(split): p_loss.detach().mean(),
                    "{}/disc_factor".format(split): torch.tensor(disc_factor),
                    "{}/g_loss".format(split): g_loss.detach().mean(),
                    "{}/mean".format(split): mean.detach().mean(),
                    "{}/std".format(split): std.detach().mean(),
                    "{}/logvar posterior".format(split): logvar.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(x.contiguous().detach())
                logits_fake = self.discriminator(xrec.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((x.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((xrec.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
