"""Noise schedule, timestep grids, reverse kernels and the one sampling loop.

Conventions
    * A model index i in [0, T-1] is a noisy level with alpha_bar[i] < 1.
    * CLEAN (= -1) is the data endpoint, alpha_bar = 1. It is never given to theta.
    * A grid is a strictly decreasing tuple of indices. Each adjacent pair (t, s)
      is one reverse edge and costs one evaluation of theta at t, so K+1 nodes
      are exactly K NFE (before CFG doubling).

Kernels
    A kernel maps (x_t, eps_theta, t, s, labels) to a ``Transition``: the mean
    and diagonal variance of p(x_s | x_t). ``sample`` draws from it and
    ``loss.RespacedVLB`` scores it, so training and inference share one
    definition of the update.

    Scaffold kernel (DDIM family, per-coordinate variance multiplier lambda):

        x0_hat  = (x_t - sqrt(1 - ab_t) eps) / sqrt(ab_t)
        beta~   = (1 - ab_s) / (1 - ab_t) * (1 - ab_t / ab_s)
        sigma^2 = min(lambda * beta~, (1 - margin) * (1 - ab_s))
        x_s     = sqrt(ab_s) x0_hat + sqrt(1 - ab_s - sigma^2) eps + sigma z

        lambda = 0   deterministic DDIM
        lambda = 1   stochastic DDIM, eta = 1 (mean = DDPM posterior mean)
        s = CLEAN    beta~ = 0: the edge returns x0_hat and injects no noise

    This is a learned finite-step sampler. With a state-dependent lambda it does
    not preserve the diffusion marginals in general; that must be measured.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

KERNEL_VERSION = 'ddim_scaffold_v1'
BUDGET_MARGIN = 1e-4
CLEAN = -1
Grid = tuple[int, ...]
EpsilonFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


# Schedule and grids

@dataclass(frozen=True)
class Schedule:
    """alpha_bar of the pretrained DDPM, float64 on CPU."""

    alpha_bar: torch.Tensor

    def __post_init__(self) -> None:
        alpha = self.alpha_bar
        if alpha.ndim != 1 or len(alpha) < 2 or alpha.dtype != torch.float64:
            raise ValueError('alpha_bar must be a float64 vector with at least two steps')
        if not bool(((alpha > 0) & (alpha < 1)).all()) or not bool((alpha[1:] < alpha[:-1]).all()):
            raise ValueError('alpha_bar must be strictly decreasing in (0, 1)')

    @classmethod
    def linear(cls, steps: int, beta_1: float, beta_T: float) -> 'Schedule':
        # float32 linspace promoted to float64, exactly as the vendor trainer builds it.
        betas = torch.linspace(beta_1, beta_T, steps).double()
        return cls(torch.cumprod(1.0 - betas, dim=0))

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'Schedule':
        diffusion = config['diffusion']
        return cls.linear(int(diffusion['steps']), float(diffusion['beta_1']), float(diffusion['beta_T']))

    def __len__(self) -> int:
        return len(self.alpha_bar)

    def alpha_bar_at(self, indices: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """alpha_bar per sample, shaped [B, 1, ...] to broadcast against ``like``; CLEAN gives 1."""
        if indices.ndim != 1 or len(indices) != len(like) or indices.is_floating_point():
            raise ValueError('indices must be an integer tensor with one entry per sample')
        if bool(((indices < CLEAN) | (indices >= len(self))).any()):
            raise ValueError('timestep index outside [CLEAN, T-1]')
        # Appending 1 lets CLEAN (= -1) index the clean endpoint directly.
        padded = torch.cat((self.alpha_bar, self.alpha_bar.new_ones(1)))
        values = padded[indices.cpu()].to(device=like.device, dtype=like.dtype)
        return values.reshape((len(like),) + (1,) * (like.ndim - 1))

    def time(self, indices: torch.Tensor) -> torch.Tensor:
        """Continuous time (index + 1) / T in (0, 1], the scaffold's time input."""
        return (indices.float() + 1.0) / len(self)


def validate_grid(grid: Grid, steps: int) -> Grid:
    grid = tuple(int(index) for index in grid)
    if len(grid) < 2 or any(a <= b for a, b in zip(grid, grid[1:])):
        raise ValueError('grid must be strictly decreasing with at least two nodes')
    if grid[0] >= steps or grid[-1] < CLEAN:
        raise ValueError('grid nodes must lie in [CLEAN, T-1]')
    return grid


def respaced_grid(steps: int, nfe: int, *, spacing: str = 'leading') -> Grid:
    """K-NFE grid ending at CLEAN.

    leading:  Song et al. DDIM spacing, evaluations at 0, T/K, 2T/K, ... The
              chain starts below T-1 and takes x0_hat from index 0, where almost
              no noise is left.
    trailing: evaluations at T-1, T-1-T/K, ... The terminal edge then jumps to
              x0_hat from a much noisier index (49 for K=20), which shrinks
              low-variance directions towards the posterior mean.
    NFE = T is the full ancestral chain T-1, ..., 0, CLEAN for either spacing.

    On the exact-score Gaussian control the DDIM family is closer to the data
    variance with ``leading``; ancestral fixedlarge prefers ``trailing``. The
    spacing is part of the protocol and is recorded with every result.
    """
    if isinstance(nfe, bool) or not isinstance(nfe, int) or not 1 <= nfe <= steps:
        raise ValueError('require integer 1 <= nfe <= T')
    if spacing == 'leading':
        nodes = (np.arange(nfe) * (steps // nfe))[::-1]
    elif spacing == 'trailing':
        nodes = np.rint(steps - 1 - np.arange(nfe) * (steps / nfe)).astype(int)
    else:
        raise ValueError("spacing must be 'leading' or 'trailing'")
    return validate_grid(tuple(int(node) for node in nodes) + (CLEAN,), steps)


def stride_grid(steps: int, stride: int) -> Grid:
    """Vendor CCUA-DDPM DDIM grid T-stride, ..., stride, 0: no clean edge, state returned at index 0."""
    if stride < 1 or steps % stride or steps // stride < 2:
        raise ValueError('stride must divide T and leave at least one edge')
    return validate_grid(tuple(range(steps - stride, -1, -stride)), steps)


def build_grid(steps: int, spec: dict[str, Any]) -> Grid:
    """Grid from a config block: {'nfe': K, 'spacing': ...} or {'stride': n}."""
    if 'stride' in spec:
        if len(spec) != 1:
            raise ValueError("a 'stride' grid takes no other keys")
        return stride_grid(steps, int(spec['stride']))
    return respaced_grid(steps, int(spec['nfe']), spacing=str(spec['spacing']))


# Transitions

@dataclass(frozen=True)
class Transition:
    """Gaussian p(x_s | x_t) with diagonal covariance.

    variance has the shape of mean and may be exactly zero (deterministic edge).
    requested is the variance asked for before the budget was applied, so
    requested / beta~ is the nominal multiplier and variance / beta~ the
    effective one. capped marks coordinates where the two differ.
    """

    x0_hat: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor
    capped: torch.Tensor
    requested: torch.Tensor

    def sample(self, noise: torch.Tensor) -> torch.Tensor:
        return self.mean + self.variance.sqrt() * noise


def posterior_variance(alpha_t: torch.Tensor, alpha_s: torch.Tensor) -> torch.Tensor:
    """Variance beta~ of q(x_s | x_t, x_0) on the respaced edge t -> s."""
    return (1.0 - alpha_s) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_s)


def scaffold_transition(x_t: torch.Tensor, epsilon: torch.Tensor, lam: torch.Tensor | float,
                        alpha_t: torch.Tensor, alpha_s: torch.Tensor) -> Transition:
    """DDIM-family edge with variance multiplier ``lam`` (scalar or per coordinate)."""
    lam = torch.as_tensor(lam, device=x_t.device, dtype=x_t.dtype)
    if not bool(torch.isfinite(lam).all()) or bool((lam < 0).any()):
        raise ValueError('the variance multiplier must be finite and nonnegative')
    requested = lam * posterior_variance(alpha_t, alpha_s)
    budget = (1.0 - BUDGET_MARGIN) * (1.0 - alpha_s)
    variance = torch.minimum(requested, budget).expand_as(x_t)
    x0_hat = (x_t - (1.0 - alpha_t).sqrt() * epsilon) / alpha_t.sqrt()
    mean = alpha_s.sqrt() * x0_hat + (1.0 - alpha_s - variance).sqrt() * epsilon
    return Transition(x0_hat, mean, variance, (requested > budget).expand_as(x_t), requested.expand_as(x_t))


def ddpm_transition(x_t: torch.Tensor, epsilon: torch.Tensor, alpha_t: torch.Tensor,
                    alpha_s: torch.Tensor, *, variance_type: str) -> Transition:
    """Ancestral DDPM edge: posterior mean with a fixed variance.

    fixedsmall uses beta~ and coincides with ``scaffold_transition(lam=1)``.
    fixedlarge uses the forward variance 1 - ab_t / ab_s. It lies outside the
    DDIM family: the mean does not compensate for the extra noise.
    """
    posterior = scaffold_transition(x_t, epsilon, 1.0, alpha_t, alpha_s)
    if variance_type == 'fixedsmall':
        variance = posterior_variance(alpha_t, alpha_s)
    elif variance_type == 'fixedlarge':
        variance = 1.0 - alpha_t / alpha_s
    else:
        raise ValueError("variance_type must be 'fixedsmall' or 'fixedlarge'")
    variance = variance.expand_as(x_t)
    return Transition(posterior.x0_hat, posterior.mean, variance, torch.zeros_like(posterior.capped), variance)


# Kernels

class ScaffoldKernel(nn.Module):
    """DDIM-family kernel whose variance multiplier is ``scaffold(x_t, time, labels)``."""

    def __init__(self, schedule: Schedule, scaffold: nn.Module) -> None:
        super().__init__()
        self.schedule = schedule
        self.scaffold = scaffold

    def forward(self, x_t: torch.Tensor, epsilon: torch.Tensor, t: torch.Tensor,
                s: torch.Tensor, labels: torch.Tensor) -> Transition:
        lam = self.scaffold(x_t, self.schedule.time(t).to(x_t.device), labels)
        return scaffold_transition(x_t, epsilon, lam, self.schedule.alpha_bar_at(t, x_t),
                                   self.schedule.alpha_bar_at(s, x_t))


class DDPMKernel(nn.Module):
    """Respaced ancestral DDPM reference with fixedsmall or fixedlarge variance."""

    def __init__(self, schedule: Schedule, *, variance_type: str = 'fixedlarge') -> None:
        super().__init__()
        if variance_type not in ('fixedsmall', 'fixedlarge'):
            raise ValueError("variance_type must be 'fixedsmall' or 'fixedlarge'")
        self.schedule = schedule
        self.variance_type = variance_type

    def forward(self, x_t: torch.Tensor, epsilon: torch.Tensor, t: torch.Tensor,
                s: torch.Tensor, labels: torch.Tensor) -> Transition:
        return ddpm_transition(x_t, epsilon, self.schedule.alpha_bar_at(t, x_t),
                               self.schedule.alpha_bar_at(s, x_t), variance_type=self.variance_type)


# Sampling loop

@torch.no_grad()
def sample(epsilon_fn: EpsilonFn, kernel: nn.Module, labels: torch.Tensor,
           initial_noise: torch.Tensor, grid: Grid, *, generator: torch.Generator,
           observer: Callable[[int, int, Transition], None] | None = None) -> torch.Tensor:
    """Run ``kernel`` down ``grid`` from ``initial_noise``; returns the unclipped endpoint.

    epsilon_fn(x, t, labels) is the frozen, already guided noise prediction.
    One noise tensor is drawn per non-terminal edge whatever the kernel does
    with it, so arms sharing ``generator`` seeds stay paired even when one of
    them is deterministic. The CLEAN edge returns the kernel mean (x0_hat).
    ``observer(t, s, transition)`` is called once per edge and must not modify it.
    """
    grid = validate_grid(grid, len(kernel.schedule))
    if labels.shape != (len(initial_noise),):
        raise ValueError('labels must contain one class per sample')
    state = initial_noise.clone()
    for current, following in zip(grid, grid[1:]):
        t = torch.full_like(labels, current)
        s = torch.full_like(labels, following)
        transition = kernel(state, epsilon_fn(state, t, labels), t, s, labels)
        if observer is not None:
            observer(current, following, transition)
        if following == CLEAN:
            state = transition.mean
        else:
            noise = torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=generator)
            state = transition.sample(noise)
        if not bool(torch.isfinite(state).all()):
            raise FloatingPointError(f'nonfinite state on edge {current}->{following}')
    return state
