"""Auditable regression adapter for an unmodified, pinned EEGain source checkout."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import torch
from torch import nn

COMMIT = "d6892f5586181f345969651a9baedabd828bb1c5"


def official_eegnet(root):
    root = Path(root).resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    tracked = ["eegain/models/eegnet.py", "eegain/models/_registry.py"]
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", *tracked], text=True)
    if commit != COMMIT or dirty.strip():
        raise ValueError("EEGain source must match the clean pinned commit")
    package = "_openaffect_eegain_v7"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(root/"eegain/models")]
        sys.modules[package] = module
    for name in ("_registry", "eegnet"):
        full = package+"."+name
        path = root/"eegain/models"/(name+".py")
        if full in sys.modules:
            if Path(sys.modules[full].__file__).resolve() != path:
                raise ValueError("Cannot switch EEGain sources in one process")
            continue
        spec = importlib.util.spec_from_file_location(full, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
    return sys.modules[package+".eegnet"].EEGNet


class EEGainRegression(nn.Module):
    """Official convolution blocks; eager, unbounded two-output regression head."""

    def __init__(self, channel_count, config):
        super().__init__()
        cls = official_eegnet(config.eegain_source)
        self.backbone = cls(num_classes=2, channels=channel_count, dropout_rate=config.dropout,
            sampling_r=config.sampling_rate, window=config.crop_samples/config.sampling_rate,
            kernel_length1=config.temporal_kernel, kernel_length2=config.separable_kernel,
            f1=config.f1, f2=config.f2, d=config.depth_multiplier)
        self.backbone.blocks.eval()
        with torch.no_grad():
            latent = self.backbone.blocks(torch.zeros(1, 1, channel_count, config.crop_samples)).flatten(1)
        self.backbone.blocks.train()
        self.head = nn.Linear(latent.shape[1], 2, bias=False)

    def forward(self, trials, *, adversary_strength=0.0):
        if adversary_strength != 0:
            raise ValueError("EEGain regression adapter has no adversarial head")
        latent = self.backbone.blocks(trials[:, None]).flatten(1)
        result = self.head(latent)
        return result, result.new_zeros((len(trials), 1)), latent
