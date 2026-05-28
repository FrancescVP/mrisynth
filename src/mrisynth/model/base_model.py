"""Abstract base class for pix2pix-style models.

Adapted from pytorch-CycleGAN-and-pix2pix (Zhu et al., 2017).
Supports single-GPU, multi-GPU DDP, and checkpoint save/load.

Expected opt fields
-------------------
  isTrain (bool)
  checkpoints_dir (str | Path)
  name (str)          – experiment name; checkpoints go to checkpoints_dir/name/
  device              – torch.device
  verbose (bool)
  init_type (str)     – weight init: normal | xavier | kaiming | orthogonal
  init_gain (float)
  continue_train (bool)
  epoch (str)         – epoch label to load, e.g. "latest" or "100"
  load_iter (int)     – if >0, load iter_{load_iter}_net_{X}.pth instead
  norm (str)          – normalisation type (used for syncbatch DDP check)
  lr_policy (str)     – linear | step | plateau | cosine
  n_epochs (int)
  n_epochs_decay (int)
  epoch_count (int)
  lr_decay_iters (int)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist

from . import networks


class BaseModel(ABC):

    def __init__(self, opt):
        self.opt        = opt
        self.isTrain    = opt.isTrain
        self.save_dir   = Path(opt.checkpoints_dir) / opt.name
        self.device     = opt.device
        torch.backends.cudnn.benchmark = True
        self.loss_names:   list[str] = []
        self.model_names:  list[str] = []
        self.visual_names: list[str] = []
        self.optimizers:   list      = []
        self.image_paths:  list      = []
        self.metric = 0  # used for lr_policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train: bool):
        return parser

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    # ------------------------------------------------------------------
    def setup(self, opt):
        """Init weights, load checkpoints, wrap DDP, build LR schedulers."""
        for name in self.model_names:
            if not isinstance(name, str):
                continue
            net = getattr(self, "net" + name)
            net = networks.init_net(net, opt.init_type, opt.init_gain)

            if not self.isTrain or opt.continue_train:
                suffix = f"iter_{opt.load_iter}" if opt.load_iter > 0 else opt.epoch
                load_path = self.save_dir / f"{suffix}_net_{name}.pth"
                if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                    net = net.module
                print(f"loading model from {load_path}")
                state_dict = torch.load(load_path, map_location=str(self.device), weights_only=True)
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata
                for key in list(state_dict.keys()):
                    self._patch_instance_norm(state_dict, net, key.split("."))
                net.load_state_dict(state_dict)

            net.to(self.device)

            if dist.is_initialized():
                net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[self.device.index])
                dist.barrier()

            setattr(self, "net" + name, net)

        self.print_networks(opt.verbose)

        if self.isTrain:
            self.schedulers = [networks.get_scheduler(opt_i, opt) for opt_i in self.optimizers]

    # ------------------------------------------------------------------
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                getattr(self, "net" + name).eval()

    def train_mode(self):
        for name in self.model_names:
            if isinstance(name, str):
                getattr(self, "net" + name).train()

    def test(self):
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        pass

    def get_image_paths(self) -> list:
        return self.image_paths

    def update_learning_rate(self):
        old_lr = self.optimizers[0].param_groups[0]["lr"]
        for scheduler in self.schedulers:
            if self.opt.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]["lr"]
        print(f"lr {old_lr:.7f} -> {lr:.7f}")

    def get_current_visuals(self) -> OrderedDict:
        ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                ret[name] = getattr(self, name)
        return ret

    def get_current_losses(self) -> OrderedDict:
        ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                val = getattr(self, "loss_" + name)
                ret[name] = val.item() if isinstance(val, torch.Tensor) else float(val)
        return ret

    def save_networks(self, epoch: str | int):
        """Save all networks to disk (rank-0 only in DDP)."""
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for name in self.model_names:
            if not isinstance(name, str):
                continue
            net = getattr(self, "net" + name)
            if hasattr(net, "module"):
                net = net.module
            if hasattr(net, "_orig_mod"):
                net = net._orig_mod
            torch.save(net.state_dict(), self.save_dir / f"{epoch}_net_{name}.pth")

    def load_networks(self, epoch: str | int):
        for name in self.model_names:
            if not isinstance(name, str):
                continue
            net = getattr(self, "net" + name)
            if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                net = net.module
            load_path = self.save_dir / f"{epoch}_net_{name}.pth"
            print(f"loading model from {load_path}")
            state_dict = torch.load(load_path, map_location=str(self.device), weights_only=True)
            if hasattr(state_dict, "_metadata"):
                del state_dict._metadata
            for key in list(state_dict.keys()):
                self._patch_instance_norm(state_dict, net, key.split("."))
            net.load_state_dict(state_dict)
        if dist.is_initialized():
            dist.barrier()

    def print_networks(self, verbose: bool = False):
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                n_params = sum(p.numel() for p in net.parameters())
                if verbose:
                    print(net)
                print(f"[Network {name}] {n_params / 1e6:.3f} M parameters")
        print("-----------------------------------------------")

    def set_requires_grad(self, nets, requires_grad: bool = False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # ------------------------------------------------------------------
    def _patch_instance_norm(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith("InstanceNorm") and key in ("running_mean", "running_var"):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and key == "num_batches_tracked":
                state_dict.pop(".".join(keys))
        else:
            self._patch_instance_norm(state_dict, getattr(module, key), keys, i + 1)
