"""3-D pix2pix model for volumetric MRI synthesis (paired).

Default task: T1w (non-contrast) -> T1CE (contrast-enhanced).

Key differences from the 2-D pix2pix model:
- 3-D U-Net generator + 3-D PatchGAN discriminator.
- No Tanh at the generator output: data is nnUNet Z-score normalised,
  not clipped to [-1, 1].
- Configurable pixel loss: L1 or L2 (MSE).
- Optional feature-matching loss (lambda_feat > 0).
- Discriminator update frequency (d_update_freq) prevents early collapse.

Expected opt fields (on top of BaseModel fields)
-------------------------------------------------
  input_nc (int)        – input channels (e.g. 1 for T1w)
  output_nc (int)       – output channels (e.g. 1 for T1CE)
  ngf (int)             – generator base filters (default 64)
  ndf (int)             – discriminator base filters (default 32)
  num_downs (int)       – U-Net depth, >= 5 (default 5)
  norm (str)            – normalisation type
  no_dropout (bool)
  init_type (str)
  init_gain (float)
  n_layers_D (int)      – PatchGAN depth (default 3)
  gan_mode (str)        – lsgan | vanilla | wgangp
  lr (float)
  beta1 (float)
  pixel_loss (str)      – "l2" (MSE, default) | "l1" (MAE)
  lambda_pixel (float)  – pixel loss weight (default 100.0)
  lambda_feat (float)   – feature-matching weight; 0 = disabled (default 10.0)
  d_update_freq (int)   – update D every N generator steps (default 2)
  direction (str)       – AtoB | BtoA
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base_model import BaseModel
from . import networks


class Pix2Pix3dModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train: bool = True):
        parser.set_defaults(
            norm="instance",
            input_nc=1,
            output_nc=1,
            ngf=64,
            ndf=32,
        )
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="lsgan")
            parser.add_argument("--pixel_loss", choices=["l1", "l2"], default="l2",
                                help="Pixel-wise reconstruction loss: l2 (MSE) or l1 (MAE).")
            parser.add_argument("--lambda_pixel", type=float, default=100.0,
                                help="Weight for the pixel-wise reconstruction loss.")
            parser.add_argument("--lambda_feat", type=float, default=10.0,
                                help="Weight for discriminator feature-matching loss. 0 = disabled.")
            parser.add_argument("--d_update_freq", type=int, default=2,
                                help="Update D once every d_update_freq generator steps.")
            parser.add_argument("--use_tumor_loss", action="store_true",
                                help="Blend G loss with ET-aware tumor loss.")
            parser.add_argument("--alpha_tumor", type=float, default=0.5,
                                help="Tumor loss blending weight (use_tumor_loss only).")
        parser.add_argument("--num_downs", type=int, default=5,
                            help="U-Net depth (>=5). 5 levels: 128->4 voxels at bottleneck.")
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names   = ["G_GAN", "G_pixel", "G_feat", "G_tumor", "D_real", "D_fake"]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        self.model_names  = ["G", "D"] if self.isTrain else ["G"]

        # 3-D U-Net generator
        self.netG = networks.define_G_3d(
            opt.input_nc, opt.output_nc, opt.ngf,
            num_downs=opt.num_downs,
            norm=opt.norm,
            use_dropout=not opt.no_dropout,
            init_type=opt.init_type,
            init_gain=opt.init_gain,
        )

        if self.isTrain:
            # Conditional discriminator: sees cat(A, G(A)) or cat(A, B)
            self.netD = networks.define_D_3d(
                opt.input_nc + opt.output_nc, opt.ndf,
                n_layers_D=opt.n_layers_D,
                norm=opt.norm,
                init_type=opt.init_type,
                init_gain=opt.init_gain,
            )

            self.criterionGAN   = networks.GANLoss(opt.gan_mode).to(self.device)
            # Pixel reconstruction loss (L2 by default)
            pixel_loss = getattr(opt, "pixel_loss", "l2")
            self.criterion_pixel: nn.Module = (
                nn.MSELoss() if pixel_loss == "l2" else nn.L1Loss()
            )
            # Feature-matching always uses L1 (features can be very large; L1 is more stable)
            self.criterion_feat = nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers += [self.optimizer_G, self.optimizer_D]

            self._d_step = 0
            # Pre-initialise so get_current_losses() never raises AttributeError
            self.loss_D_fake  = torch.tensor(0.0)
            self.loss_D_real  = torch.tensor(0.0)
            self.loss_G_feat  = torch.tensor(0.0)
            self.loss_G_tumor = torch.tensor(0.0)

            if getattr(opt, "use_tumor_loss", False):
                from ..losses import make_default_et_aware_loss
                self._tumor_loss_fn = make_default_et_aware_loss(alpha=opt.alpha_tumor)

    # ------------------------------------------------------------------
    def set_input(self, batch: dict):
        """Unpack a batch dict produced by Pix2Pix3dDataset."""
        AtoB = self.opt.direction == "AtoB"
        self.real_A = batch["A" if AtoB else "B"].to(self.device)
        self.real_B = batch["B" if AtoB else "A"].to(self.device)
        self.image_paths = batch.get("A_paths" if AtoB else "B_paths", [])
        seg = batch.get("seg")
        self.seg = seg.to(self.device) if seg is not None else None

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    # ------------------------------------------------------------------
    def backward_D(self):
        # Fake pair: cat(A, G(A)) — D should say "fake"
        fake_AB   = torch.cat((self.real_A, self.fake_B), dim=1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real pair: cat(A, B) — D should say "real"
        real_AB   = torch.cat((self.real_A, self.real_B), dim=1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), dim=1)

        if self.opt.lambda_feat > 0:
            # Single D forward to get both features and prediction.
            # Bypass DDP wrapper: DDP doesn't handle tuple outputs natively.
            d_net = self.netD.module if hasattr(self.netD, "module") else self.netD
            fake_feats, pred_fake = d_net(fake_AB, return_features=True)
            real_AB = torch.cat((self.real_A, self.real_B), dim=1)
            with torch.no_grad():
                real_feats, _ = d_net(real_AB, return_features=True)
            self.loss_G_feat = sum(
                self.criterion_feat(ff, rf)
                for ff, rf in zip(fake_feats, real_feats)
            ) * self.opt.lambda_feat
        else:
            pred_fake = self.netD(fake_AB)
            self.loss_G_feat = torch.tensor(0.0)

        self.loss_G_GAN   = self.criterionGAN(pred_fake, True)
        self.loss_G_pixel = self.criterion_pixel(self.fake_B, self.real_B) * self.opt.lambda_pixel
        self.loss_G = self.loss_G_GAN + self.loss_G_pixel + self.loss_G_feat

        if hasattr(self, "_tumor_loss_fn") and self.seg is not None:
            self.loss_G, info = self._tumor_loss_fn(
                self.loss_G, self.fake_B, self.real_B, self.seg
            )
            t = info.get("tumor_loss")
            self.loss_G_tumor = torch.tensor(t if t is not None else 0.0)
        else:
            self.loss_G_tumor = torch.tensor(0.0)

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self._d_step += 1

        # Update D every d_update_freq generator steps
        if self._d_step % self.opt.d_update_freq == 0:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # Always update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
