"""Small mathematical core for the frozen-score scalar scaffold.

The score network is supplied by the caller and remains frozen. Only
``ScalarScaffold`` is trainable.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ScalarScaffold(nn.Module):
    """Learn rho(y, t), initialized exactly at the native SDE value rho=1."""

    def __init__(
        self,
        *,
        hidden_width: int = 32,
        rho_min: float = 0.05,
        rho_max: float = 2.0,
    ) -> None:
        super().__init__()
        if not 0.0 < rho_min < 1.0 < rho_max:
            raise ValueError("rho bounds must satisfy 0 < rho_min < 1 < rho_max")
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.network = nn.Sequential(
            nn.Linear(2, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1),
        )
        output = self.network[-1]
        nn.init.zeros_(output.weight)
        target = (1.0 - self.rho_min) / (self.rho_max - self.rho_min)
        nn.init.constant_(output.bias, math.log(target / (1.0 - target)))

    def forward(
        self,
        normalized_time: torch.Tensor,
        normalized_log_count: torch.Tensor,
    ) -> torch.Tensor:
        time, log_count = torch.broadcast_tensors(
            normalized_time, normalized_log_count
        )
        features = torch.stack((time, log_count), dim=-1)
        raw = self.network(features).squeeze(-1)
        return self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(raw)


def alpha_bar_schedule(
    steps: int,
    beta_1: float,
    beta_t: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    betas = torch.linspace(beta_1, beta_t, steps, device=device)
    return torch.cumprod(1.0 - betas, dim=0)


def segments(steps: int, stride: int) -> list[tuple[int, int]]:
    if steps < 1 or stride < 1:
        raise ValueError("steps and stride must be positive")
    return [(start, min(start + stride, steps)) for start in range(0, steps, stride)]


def normalized_log_counts(counts: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    scale = torch.log(counts.max().float()).clamp_min(1.0)
    return torch.log(counts.index_select(0, labels).float().clamp_min(1.0)) / scale


def predict_epsilon(
    model: nn.Module,
    images: torch.Tensor,
    times: torch.Tensor,
    labels: torch.Tensor,
    *,
    microbatch: int,
    preserve_input_gradient: bool,
    guidance_scale: float = 0.0,
) -> torch.Tensor:
    """Evaluate frozen theta with the CFG convention used by CCUA-DDPM.

    ``guidance_scale`` is the vendor sampler's ``omega``: the guided prediction
    is ``eps_cond + omega * (eps_cond - eps_uncond)``.  A zero value avoids the
    extra unconditional model pass.
    """
    if microbatch < 1:
        raise ValueError("microbatch must be positive")
    if not math.isfinite(guidance_scale):
        raise ValueError("guidance_scale must be finite")
    predictions = []

    def call_conditional(x, t, y):
        return model(x, t, y=y)

    def call_unconditional(x, t):
        return model(x, t, y=None)

    for start in range(0, len(images), microbatch):
        stop = start + microbatch
        x = images[start:stop]
        t = times[start:stop]
        y = labels[start:stop]
        if preserve_input_gradient and x.requires_grad:
            conditional = checkpoint(call_conditional, x, t, y, use_reentrant=False)
        else:
            conditional = call_conditional(x, t, y)
        if guidance_scale == 0.0:
            prediction = conditional
        else:
            if preserve_input_gradient and x.requires_grad:
                unconditional = checkpoint(call_unconditional, x, t, use_reentrant=False)
            else:
                unconditional = call_unconditional(x, t)
            prediction = conditional + guidance_scale * (conditional - unconditional)
        predictions.append(prediction)
    return torch.cat(predictions, dim=0)


def forward_path_pair(
    model: nn.Module,
    scaffold: ScalarScaffold,
    clean_images: torch.Tensor,
    labels: torch.Tensor,
    counts: torch.Tensor,
    alpha_bar: torch.Tensor,
    *,
    steps: int,
    stride: int,
    microbatch: int,
    generator: torch.Generator | None = None,
    guidance_scale: float = 0.0,
) -> torch.Tensor:
    """Produce two independent scaffold paths for each clean source image."""
    batch_size = len(clean_images)
    images = torch.cat((clean_images, clean_images), dim=0)
    path_labels = labels.repeat(2)
    log_counts = normalized_log_counts(counts, path_labels)
    one_step_alpha = torch.cat((alpha_bar[:1], alpha_bar[1:] / alpha_bar[:-1]))

    for low, high in segments(steps, stride):
        segment_alpha = float(one_step_alpha[low:high].prod().detach().cpu())
        sqrt_alpha = math.sqrt(segment_alpha)
        time = torch.full(
            (len(images),),
            (low + high) / (2.0 * steps),
            dtype=images.dtype,
            device=images.device,
        )
        rho = scaffold(time, log_counts).to(images.dtype).view(-1, 1, 1, 1)
        model_time = torch.full(
            (len(images),), low, dtype=torch.long, device=images.device
        )
        epsilon = predict_epsilon(
            model,
            images,
            model_time,
            path_labels,
            microbatch=microbatch,
            preserve_input_gradient=True,
            guidance_scale=guidance_scale,
        )
        sigma = (1.0 - alpha_bar[model_time]).clamp_min(1e-8).sqrt()
        score = -epsilon / sigma.view(-1, 1, 1, 1)
        noise = torch.randn(
            images.shape,
            dtype=images.dtype,
            device=images.device,
            generator=generator,
        )
        images = (
            sqrt_alpha * images
            + (1.0 - sqrt_alpha) * (rho - 1.0) * score
            + math.sqrt(max(0.0, 1.0 - segment_alpha)) * rho.sqrt() * noise
        )
    return images.view(2, batch_size, *images.shape[1:])


def reverse_sample(
    model: nn.Module,
    scaffold: ScalarScaffold | None,
    labels: torch.Tensor,
    counts: torch.Tensor,
    alpha_bar: torch.Tensor,
    *,
    image_size: int,
    steps: int,
    stride: int,
    microbatch: int,
    generator: torch.Generator,
    rho_override: float | None = None,
    rho_window: tuple[float, float] | None = None,
    guidance_scale: float = 0.0,
) -> torch.Tensor:
    """Sample the generalized DDIM family with variance ratio ``rho``.

    For a jump from timestep ``t`` to ``s``, ``rho`` multiplies the DDIM
    posterior variance.  Thus ``rho=0`` is exactly the deterministic DDIM
    update used as the PF-ODE discretization, while ``rho=1`` recovers the
    standard DDPM posterior variance for adjacent steps.  A finite jump can
    make ``rho>1`` request more variance than the target marginal contains;
    only in that case the variance is projected to the valid boundary.
    """
    if scaffold is None and rho_override is None:
        raise ValueError("provide scaffold or rho_override")
    if rho_window is not None:
        lower, upper = rho_window
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError("rho_window must satisfy 0 <= lower < upper <= 1")
        if rho_override is None:
            raise ValueError("rho_window requires a fixed rho_override")
    images = torch.randn(
        (len(labels), 3, image_size, image_size),
        device=labels.device,
        generator=generator,
    )
    log_counts = normalized_log_counts(counts, labels)
    for current_step in reversed(range(0, steps, stride)):
        # The vendor DDIM loop's t=0 update is the identity.
        if current_step == 0:
            continue
        next_step = max(current_step - stride, 0)
        time = torch.full(
            (len(images),),
            current_step / float(steps),
            dtype=images.dtype,
            device=images.device,
        )
        if rho_override is None:
            rho = scaffold(time, log_counts).to(images.dtype)
        else:
            rho = torch.full_like(time, float(rho_override))
        if rho_window is not None:
            lower, upper = rho_window
            active = (time >= lower) & (time < upper)
            rho = torch.where(active, rho, torch.zeros_like(rho))
        rho = rho.view(-1, 1, 1, 1)
        model_time = torch.full(
            (len(images),), current_step, dtype=torch.long, device=images.device
        )
        epsilon = predict_epsilon(
            model,
            images,
            model_time,
            labels,
            microbatch=microbatch,
            preserve_input_gradient=False,
            guidance_scale=guidance_scale,
        )
        current_alpha = alpha_bar[current_step].to(dtype=images.dtype)
        next_alpha = alpha_bar[next_step].to(dtype=images.dtype)
        predicted_clean = (
            images - (1.0 - current_alpha).sqrt() * epsilon
        ) / current_alpha.sqrt()
        base_variance = (
            (1.0 - next_alpha) / (1.0 - current_alpha)
            * (1.0 - current_alpha / next_alpha)
        ).clamp_min(0.0)
        requested_variance = rho * base_variance
        maximum_variance = (1.0 - next_alpha).clamp_min(0.0)
        variance = torch.minimum(requested_variance.clamp_min(0.0), maximum_variance)
        direction_variance = (1.0 - next_alpha - variance).clamp_min(0.0)
        noise = torch.randn(
            images.shape,
            dtype=images.dtype,
            device=images.device,
            generator=generator,
        )
        images = (
            next_alpha.sqrt() * predicted_clean
            + direction_variance.sqrt() * epsilon
            + variance.sqrt() * noise
        )
    return images


def effective_class_weights(counts: torch.Tensor, exponent: float) -> torch.Tensor:
    probability = counts.float() / counts.sum()
    weights = probability * (counts.max() / counts).pow(exponent)
    return weights / weights.mean()


def budget_loss(
    scaffold: ScalarScaffold,
    counts: torch.Tensor,
    *,
    time_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = counts / counts.sum()
    times = torch.rand((time_samples, 1), device=counts.device)
    class_counts = torch.log(counts.clamp_min(1.0)) / torch.log(counts.max()).clamp_min(1.0)
    rho = scaffold(times.expand(-1, len(counts)), class_counts.unsqueeze(0))
    weighted_rho = (rho * probabilities.unsqueeze(0)).sum(dim=1)
    return (weighted_rho - 1.0).square().mean(), rho.mean()


def sliced_wasserstein(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    projections: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    first = first.flatten(1)
    second = second.flatten(1)
    if first.shape != second.shape:
        raise ValueError(f"SWD shapes differ: {first.shape} and {second.shape}")
    directions = torch.randn(
        (projections, first.shape[1]),
        device=first.device,
        generator=generator,
    )
    directions = F.normalize(directions, dim=1)
    first_projection = (first @ directions.T).sort(dim=0).values
    second_projection = (second @ directions.T).sort(dim=0).values
    return (first_projection - second_projection).abs().mean()
