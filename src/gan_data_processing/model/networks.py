"""3-D network architectures for volumetric MRI synthesis.

Adapted from pytorch-CycleGAN-and-pix2pix (Zhu et al., 2017).
Only the 3-D components are kept: 2-D generators and discriminators are not
included since this package targets volumetric (nnUNet-preprocessed) data.
"""
from __future__ import annotations

import functools
import os

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer_3d(norm_type: str = "instance"):
    """Return a 3-D normalisation layer factory."""
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == "batch":
        return functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == "syncbatch":
        return functools.partial(nn.SyncBatchNorm, affine=True, track_running_stats=True)
    elif norm_type == "none":
        return lambda x: Identity()
    else:
        raise NotImplementedError(f"normalisation layer [{norm_type}] not found")


def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02):
    """Initialise network weights in-place."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"init method [{init_type}] not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print(f"initialize network with {init_type}")
    net.apply(init_func)


def init_net(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> nn.Module:
    """Move network to GPU (if available) and initialise weights."""
    if torch.cuda.is_available():
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            net.to(local_rank)
        else:
            net.to(0)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    """Return a learning-rate scheduler.

    opt must expose: lr_policy, n_epochs, n_epochs_decay, epoch_count,
    lr_decay_iters.
    """
    if opt.lr_policy == "linear":
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        return lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == "plateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        raise NotImplementedError(f"lr policy [{opt.lr_policy}] not implemented")


# ---------------------------------------------------------------------------
# GAN loss
# ---------------------------------------------------------------------------

class GANLoss(nn.Module):
    """Wraps GAN objective so callers don't need to build target tensors.

    Supports: lsgan | vanilla | wgangp.
    """

    def __init__(self, gan_mode: str, target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "wgangp":
            self.loss = None
        else:
            raise NotImplementedError(f"gan mode [{gan_mode}] not implemented")

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = self.real_label if target_is_real else self.fake_label
        return target.expand_as(prediction)

    def __call__(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if self.gan_mode in ("lsgan", "vanilla"):
            return self.loss(prediction, self.get_target_tensor(prediction, target_is_real))
        elif self.gan_mode == "wgangp":
            return -prediction.mean() if target_is_real else prediction.mean()


# ---------------------------------------------------------------------------
# 3-D factory functions
# ---------------------------------------------------------------------------

def define_G_3d(
    input_nc: int,
    output_nc: int,
    ngf: int,
    num_downs: int = 5,
    norm: str = "instance",
    use_dropout: bool = False,
    init_type: str = "normal",
    init_gain: float = 0.02,
) -> nn.Module:
    """Create a 3-D U-Net generator.

    Weight init and device placement are handled by BaseModel.setup().
    """
    norm_layer = get_norm_layer_3d(norm)
    return UnetGenerator3D(input_nc, output_nc, num_downs, ngf,
                           norm_layer=norm_layer, use_dropout=use_dropout)


def define_D_3d(
    input_nc: int,
    ndf: int,
    n_layers_D: int = 3,
    norm: str = "instance",
    init_type: str = "normal",
    init_gain: float = 0.02,
) -> nn.Module:
    """Create a 3-D PatchGAN discriminator."""
    norm_layer = get_norm_layer_3d(norm)
    return NLayerDiscriminator3D(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer)


# ---------------------------------------------------------------------------
# 3-D U-Net generator
# ---------------------------------------------------------------------------

class UnetGenerator3D(nn.Module):
    """3-D U-Net generator built from nested UnetSkipConnectionBlock3D modules."""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False):
        super().__init__()
        # Build from innermost outward (same recursive pattern as 2-D UnetGenerator)
        unet_block = UnetSkipConnectionBlock3D(
            ngf * 8, ngf * 8, submodule=None, norm_layer=norm_layer, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock3D(
                ngf * 8, ngf * 8, submodule=unet_block,
                norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock3D(ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3D(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3D(ngf,     ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock3D(
            output_nc, ngf, input_nc=input_nc,
            submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetSkipConnectionBlock3D(nn.Module):
    """One encoder-decoder level of the 3-D U-Net with a skip connection.

    X ──────────────────────────────────────── identity ──────────────────┐
    └── downconv ── [norm] ── [submodule] ── upconv ── [norm] ──  cat ──►
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False):
        super().__init__()
        self.outermost = outermost

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func in (nn.InstanceNorm3d, nn.InstanceNorm2d)
        else:
            use_bias = norm_layer in (nn.InstanceNorm3d, nn.InstanceNorm2d)

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu   = nn.ReLU(True)
        upnorm   = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            # No Tanh: data is Z-score normalised, not bounded to [-1, 1]
            model = [downconv, submodule, uprelu, upconv]
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up   = [uprelu, upconv, upnorm]
            model = down + [submodule] + up + ([nn.Dropout(0.5)] if use_dropout else [])

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        y = self.model(x)
        # Trim x to y's spatial dims before concatenating.
        # Needed when any spatial dim is odd: Conv3d floor-divides while
        # ConvTranspose3d multiplies, so the upsampled size can be input-1.
        x = x[:, :, : y.shape[2], : y.shape[3], : y.shape[4]]
        return torch.cat([x, y], dim=1)


# ---------------------------------------------------------------------------
# 3-D PatchGAN discriminator
# ---------------------------------------------------------------------------

class NLayerDiscriminator3D(nn.Module):
    """3-D PatchGAN discriminator (conditional: receives cat(A, B) as input).

    Layers are stored as a ModuleList so that intermediate feature maps can be
    extracted for feature-matching loss computation.
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm3d):
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func in (nn.InstanceNorm3d, nn.InstanceNorm2d)
        else:
            use_bias = norm_layer in (nn.InstanceNorm3d, nn.InstanceNorm2d)

        kw, padw = 4, 1
        blocks: list[nn.Module] = []

        # Block 0: first strided conv, no norm
        blocks.append(nn.Sequential(
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ))

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            blocks.append(nn.Sequential(
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        blocks.append(nn.Sequential(
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ))

        self.blocks = nn.ModuleList(blocks)
        self.final  = nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        out = self.final(x)
        if return_features:
            return features, out
        return out
