"""Networks: the frozen denoiser theta and the learned scaffold phi.

Every scaffold has the same call signature,

    scaffold(state, time, labels) -> lambda >= 0, broadcastable to state,

where ``time`` is (index+1)/T in (0, 1]. ``sampling.ScaffoldKernel`` turns
lambda into the per-coordinate variance of a reverse step. Scaffolds never see
theta or its prediction.
"""
from __future__ import annotations

import math
from importlib import import_module
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from scaffold.utils import rooted_path, sha256_file


# Frozen denoiser theta

def load_frozen_score(config: dict[str, Any], device: torch.device) -> tuple[nn.Module, str]:
    """Load the pretrained EMA UNet, frozen and in eval mode, and return its SHA-256.

    The file must match ``paths.checkpoint_sha256``; a different score is never
    silently substituted.
    """
    path = rooted_path(config['paths']['checkpoint'])
    digest = sha256_file(path)
    if digest != config['paths']['checkpoint_sha256']:
        raise RuntimeError(f'frozen score SHA mismatch: {path}')
    settings = config['model']
    unet = import_module('third_party.CCUA-DDPM.model.model').UNet
    model = unet(T=int(config['diffusion']['steps']), ch=int(settings['channels']),
                 ch_mult=list(settings['channel_multipliers']), attn=list(settings['attention_levels']),
                 num_res_blocks=int(settings['residual_blocks']), dropout=float(settings['dropout']),
                 cond=True, num_class=int(config['dataset']['num_classes']))
    weights = torch.load(path, map_location='cpu', weights_only=False)[settings['checkpoint_key']]
    model.load_state_dict({name.removeprefix('module.'): value for name, value in weights.items()}, strict=True)
    return model.requires_grad_(False).to(device).eval(), digest


class FrozenEpsilon:
    """Frozen theta as epsilon_fn(x, t, labels), with the vendor CFG convention.

    eps = eps_cond + guidance_scale * (eps_cond - eps_uncond); a scale of 0
    skips the unconditional pass. No graph is built: phi never needs d eps / d x.
    """

    def __init__(self, model: nn.Module, *, guidance_scale: float, microbatch: int) -> None:
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError('theta must be frozen and in eval mode')
        if not math.isfinite(guidance_scale) or microbatch < 1:
            raise ValueError('guidance_scale must be finite and microbatch positive')
        self.model = model
        self.guidance_scale = float(guidance_scale)
        self.microbatch = int(microbatch)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        predictions = []
        for start in range(0, len(x), self.microbatch):
            chunk = slice(start, start + self.microbatch)
            epsilon = self.model(x[chunk], t[chunk], y=labels[chunk])
            if self.guidance_scale != 0.0:
                epsilon = epsilon + self.guidance_scale * (epsilon - self.model(x[chunk], t[chunk], y=None))
            predictions.append(epsilon)
        return torch.cat(predictions).to(x.dtype)


# Learned scaffold phi

class _Conditioning(nn.Module):
    """Embedding of (time, class) shared by the learned scaffolds."""

    def __init__(self, num_classes: int, dim: int) -> None:
        super().__init__()
        self.time = nn.Sequential(nn.Linear(6, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.label = nn.Embedding(num_classes, dim)

    def forward(self, time: torch.Tensor | float, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clock = torch.as_tensor(time, device=labels.device, dtype=self.label.weight.dtype).expand(len(labels))
        features = torch.stack((clock, clock.square(),
                                torch.sin(math.pi * clock), torch.cos(math.pi * clock),
                                torch.sin(2 * math.pi * clock), torch.cos(2 * math.pi * clock)), dim=1)
        return self.time(features), self.label(labels)


class _PositiveHead(nn.Module):
    """lambda = floor + softplus(raw + bias), equal to ``initial`` when raw = 0."""

    def __init__(self, floor: float, initial: float) -> None:
        super().__init__()
        if not 0 < floor < initial or not math.isfinite(initial):
            raise ValueError('require 0 < variance_floor < variance_initial')
        self.floor = float(floor)
        self.bias = math.log(math.expm1(initial - floor))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.floor + F.softplus(raw + self.bias)


class MLPScaffold(nn.Module):
    """Flattened state + time + class -> one multiplier per pixel. No spatial prior."""

    def __init__(self, num_classes: int, *, image_size: int = 32, image_channels: int = 3,
                 hidden_dims: tuple[int, ...] = (128, 128), condition_dim: int = 128,
                 variance_floor: float = 1e-4, variance_initial: float = 1.0) -> None:
        super().__init__()
        if not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError('hidden_dims must be a non-empty sequence of positive widths')
        dimensions = image_channels * image_size ** 2
        self.condition = _Conditioning(num_classes, condition_dim)
        widths = (dimensions + 2 * condition_dim, *hidden_dims)
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths, widths[1:]))
        self.output = nn.Linear(widths[-1], dimensions)
        self.head = _PositiveHead(variance_floor, variance_initial)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state: torch.Tensor, time: torch.Tensor | float, labels: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat((state.flatten(1), *self.condition(time, labels)), dim=1)
        for layer in self.hidden:
            hidden = F.silu(layer(hidden))
        return self.head(self.output(hidden)).reshape_as(state)


class _ResBlock(nn.Module):
    """Two 3x3 convolutions with FiLM conditioning: h <- h * (1 + scale) + shift."""

    def __init__(self, input_channels: int, output_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, output_channels)
        self.film = nn.Linear(condition_dim, 2 * output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = (nn.Identity() if input_channels == output_channels
                     else nn.Conv2d(input_channels, output_channels, 1))

    def forward(self, images: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(images)))
        scale, shift = self.film(condition).chunk(2, dim=1)
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(images) + self.conv2(F.silu(hidden))


class UNetScaffold(nn.Module):
    """Small conditional U-Net: state + time + class -> one multiplier per pixel."""

    def __init__(self, num_classes: int, *, image_channels: int = 3, base_channels: int = 32,
                 channel_multipliers: tuple[int, ...] = (1, 2, 4), residual_blocks: int = 2,
                 condition_dim: int = 128, variance_floor: float = 1e-4, variance_initial: float = 1.0) -> None:
        super().__init__()
        if base_channels % 8 or not channel_multipliers or residual_blocks < 1:
            raise ValueError('base_channels must be divisible by 8, with at least one level and block')
        self.condition = _Conditioning(num_classes, condition_dim)
        channels = [base_channels * multiplier for multiplier in channel_multipliers]

        def blocks(first_input: int, width: int) -> nn.ModuleList:
            return nn.ModuleList(_ResBlock(first_input if index == 0 else width, width, condition_dim)
                                 for index in range(residual_blocks))

        self.stem = nn.Conv2d(image_channels, channels[0], 3, padding=1)
        self.down = nn.ModuleList(blocks(width, width) for width in channels)
        self.downsample = nn.ModuleList(nn.Conv2d(a, b, 3, stride=2, padding=1)
                                        for a, b in zip(channels, channels[1:]))
        self.middle = blocks(channels[-1], channels[-1])
        decoder = list(reversed(channels))
        self.upsample = nn.ModuleList(nn.Conv2d(a, b, 3, padding=1) for a, b in zip(decoder, decoder[1:]))
        self.up = nn.ModuleList(blocks(2 * width, width) for width in decoder[1:])
        self.output_norm = nn.GroupNorm(8, channels[0])
        self.output = nn.Conv2d(channels[0], image_channels, 3, padding=1)
        self.head = _PositiveHead(variance_floor, variance_initial)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state: torch.Tensor, time: torch.Tensor | float, labels: torch.Tensor) -> torch.Tensor:
        condition = sum(self.condition(time, labels))
        hidden = self.stem(state)
        skips = []
        for level, level_blocks in enumerate(self.down):
            for block in level_blocks:
                hidden = block(hidden, condition)
            skips.append(hidden)
            if level < len(self.downsample):
                hidden = self.downsample[level](hidden)
        for block in self.middle:
            hidden = block(hidden, condition)
        for upsample, level_blocks, skip in zip(self.upsample, self.up, reversed(skips[:-1])):
            hidden = upsample(F.interpolate(hidden, size=skip.shape[-2:], mode='nearest'))
            hidden = torch.cat((hidden, skip), dim=1)
            for block in level_blocks:
                hidden = block(hidden, condition)
        return self.head(self.output(F.silu(self.output_norm(hidden))))


# Fixed and control scaffolds

class FixedScaffold(nn.Module):
    """Constant lambda: 0 is deterministic DDIM, 1 is stochastic DDIM with eta = 1."""

    def __init__(self, value: float) -> None:
        super().__init__()
        if not math.isfinite(value) or value < 0:
            raise ValueError('fixed lambda must be finite and nonnegative')
        self.value = float(value)

    def forward(self, state: torch.Tensor, time: torch.Tensor | float, labels: torch.Tensor) -> torch.Tensor:
        return state.new_full((len(state),) + (1,) * (state.ndim - 1), self.value)


class TimeOnlyScaffold(nn.Module):
    """Hide the state from ``scaffold`` so that lambda depends on (time, class) only.

    The same wrapper is used to train and to sample the time-only control.
    """

    def __init__(self, scaffold: nn.Module) -> None:
        super().__init__()
        self.scaffold = scaffold

    def forward(self, state: torch.Tensor, time: torch.Tensor | float, labels: torch.Tensor) -> torch.Tensor:
        return self.scaffold(torch.zeros_like(state), time, labels)


class ShuffledStateScaffold(nn.Module):
    """Inference-only control: each sample is given the state of another sample of its class.

    States are rotated by one within each class, so lambda keeps its
    per-timestep, per-class distribution but loses its alignment with the state
    it acts on. Every class present in the batch needs at least two samples.
    """

    def __init__(self, scaffold: nn.Module) -> None:
        super().__init__()
        self.scaffold = scaffold

    def forward(self, state: torch.Tensor, time: torch.Tensor | float, labels: torch.Tensor) -> torch.Tensor:
        donor = torch.arange(len(state), device=state.device)
        for class_id in labels.unique():
            members = (labels == class_id).nonzero().flatten()
            if len(members) < 2:
                raise ValueError('the shuffled-state control needs at least two samples of every class in a batch')
            donor[members] = members.roll(1)
        return self.scaffold(state[donor], time, labels)


def build_scaffold(settings: dict[str, Any], num_classes: int) -> nn.Module:
    """Learned scaffold from the ``scaffold`` config block."""
    head = {'variance_floor': float(settings['variance_floor']),
            'variance_initial': float(settings['variance_initial']),
            'condition_dim': int(settings['condition_dim'])}
    kind = settings['kind']
    if kind == 'mlp':
        scaffold: nn.Module = MLPScaffold(num_classes, image_size=int(settings['image_size']),
                                          hidden_dims=tuple(settings['hidden_dims']), **head)
    elif kind == 'unet':
        scaffold = UNetScaffold(num_classes, base_channels=int(settings['base_channels']),
                                channel_multipliers=tuple(settings['channel_multipliers']),
                                residual_blocks=int(settings['residual_blocks']), **head)
    else:
        raise ValueError("scaffold.kind must be 'mlp' or 'unet'")
    inputs = settings['inputs']
    if inputs == 'time_class':
        return TimeOnlyScaffold(scaffold)
    if inputs != 'state_time_class':
        raise ValueError("scaffold.inputs must be 'state_time_class' or 'time_class'")
    return scaffold
