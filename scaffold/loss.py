"""Objectives for phi, scored on the sampler's own ``Transition`` objects.

A loss never re-derives the reverse update: it asks the kernel for
p(x_s | x_t) and compares that Gaussian with a target. Changing the loss or
the scaffold architecture therefore leaves sampling.py untouched.

Respaced variational bound for a grid t_K > ... > t_1 > CLEAN:

    -log p(x0) <= prior + sum_{k>=2} KL(q(x_{t_{k-1}} | x_{t_k}, x0) || p_phi(. | x_{t_k})) + recon
    recon = -log p(x0 | x_{t_1})

q is the fixed forward process restricted to the grid, so each term needs only
x_t ~ q(x_t | x0); the intermediate states are integrated out exactly.
(Conditioning the path on its noise endpoint as well gives the same KL terms,
because q(x_s | x_t, x0) does not depend on x_T.)

The terminal edge is deterministic (x0_hat) and has no Gaussian density, so the
bound uses a discretized-Gaussian decoder centred on that same x0_hat with a
fixed variance. recon and prior are constant in phi: they complete the reported
bound but carry no training signal.

Objective interface
    An objective is built by ``build_loss`` from the ``loss`` config block and
    is called as ``objective(epsilon_fn, kernel, x0, labels, generator=...)``,
    returning a scalar to minimise and a dict of detached details. It talks to
    the sampler only through ``kernel``; ``register_objective`` adds a new one
    without touching sampling.py.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch
from torch import nn

from scaffold.sampling import (
    CLEAN, EpsilonFn, Grid, Schedule, posterior_variance, scaffold_transition, validate_grid,
)

DIAGNOSTIC_LAMBDAS = (0.05, 0.25, 0.5, 1.0, 1.5, 2.0)


class Objective(Protocol):
    """What training needs from an objective; see the module docstring."""

    def __call__(self, epsilon_fn: EpsilonFn, kernel: nn.Module, x0: torch.Tensor, labels: torch.Tensor, *,
                 generator: torch.Generator) -> tuple[torch.Tensor, dict[str, Any]]: ...


ObjectiveFactory = Callable[[dict[str, Any], Schedule, Grid], Objective]
OBJECTIVES: dict[str, ObjectiveFactory] = {}


def register_objective(kind: str) -> Callable[[ObjectiveFactory], ObjectiveFactory]:
    """Make ``loss.kind = kind`` build the decorated factory(settings, schedule, grid)."""
    def decorator(factory: ObjectiveFactory) -> ObjectiveFactory:
        if kind in OBJECTIVES:
            raise ValueError(f'objective {kind!r} is already registered')
        OBJECTIVES[kind] = factory
        return factory
    return decorator


def _per_sample(value: torch.Tensor) -> torch.Tensor:
    return value.flatten(1).sum(dim=1)


def gaussian_kl(mean_q: torch.Tensor, variance_q: torch.Tensor,
                mean_p: torch.Tensor, variance_p: torch.Tensor) -> torch.Tensor:
    """KL(N(mean_q, variance_q) || N(mean_p, variance_p)) in nats, summed per sample."""
    if bool((variance_q <= 0).any()) or bool((variance_p <= 0).any()):
        raise ValueError('Gaussian KL needs strictly positive variances')
    ratio = variance_q / variance_p
    return _per_sample(0.5 * (ratio - 1.0 - ratio.log() + (mean_q - mean_p).square() / variance_p))


def discretized_gaussian_log_likelihood(x0: torch.Tensor, mean: torch.Tensor,
                                        variance: torch.Tensor) -> torch.Tensor:
    """log p(x0) for 8-bit pixels scaled to [-1, 1], summed per sample (DDPM decoder)."""
    scale = variance.sqrt()
    normal = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
    upper = normal.cdf((x0 - mean + 1.0 / 255.0) / scale)
    lower = normal.cdf((x0 - mean - 1.0 / 255.0) / scale)
    # The two edge bins absorb the tails so the 256 bin probabilities sum to one.
    probability = torch.where(x0 < -0.999, upper, torch.where(x0 > 0.999, 1.0 - lower, upper - lower))
    return _per_sample(probability.clamp_min(1e-12).log())


def forward_sample(x0: torch.Tensor, alpha: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """x_t ~ q(x_t | x0)."""
    return alpha.sqrt() * x0 + (1.0 - alpha).sqrt() * noise


def forward_posterior(x0: torch.Tensor, x_t: torch.Tensor, alpha_t: torch.Tensor,
                      alpha_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and variance of q(x_s | x_t, x0) on the respaced edge t -> s."""
    step = alpha_t / alpha_s
    mean = (alpha_s.sqrt() * (1.0 - step) * x0 + step.sqrt() * (1.0 - alpha_s) * x_t) / (1.0 - alpha_t)
    return mean, posterior_variance(alpha_t, alpha_s)


class RespacedVLB:
    """Variational bound of the K-step sampler defined by ``grid``.

    Reference and diagnostic objective. For a given mean, each KL term is
    minimised by the variance beta~ + squared mean error >= beta~, and in every
    control tested no edge preferred lambda < 1, so this bound cannot show a
    benefit that needs lambda < 1; see CAVEAT.
    """

    CAVEAT = ('respaced_vlb favours lambda >= 1 (for a given mean the optimal variance is beta~ + squared mean '
              'error) and its largest term can sit on an edge whose variance is clipped; it is a reference '
              'objective, not evidence for or against adaptive lambda < 1')

    def __init__(self, schedule: Schedule, grid: Grid) -> None:
        grid = validate_grid(grid, len(schedule))
        if grid[-1] != CLEAN or len(grid) < 3:
            raise ValueError('the bound needs a grid ending at CLEAN with at least one noisy edge')
        self.schedule = schedule
        self.grid = grid
        self.edge_t = torch.tensor(grid[:-2], dtype=torch.long)
        self.edge_s = torch.tensor(grid[1:-1], dtype=torch.long)

    @property
    def num_edges(self) -> int:
        """Number of stochastic (non-terminal) edges."""
        return len(self.edge_t)

    def stratified_edges(self, batch: int, generator: torch.Generator) -> torch.Tensor:
        """One edge per sample: uniform marginally, evenly spread within the batch."""
        device = generator.device
        offset = torch.randint(self.num_edges, (1,), generator=generator, device=device)
        order = torch.randperm(batch, generator=generator, device=device)
        return ((torch.arange(batch, device=device) + offset) % self.num_edges)[order]

    def edge_kl(self, epsilon_fn: EpsilonFn, kernel: nn.Module, x0: torch.Tensor,
                labels: torch.Tensor, edges: torch.Tensor, noise: torch.Tensor) -> dict[str, torch.Tensor]:
        """KL term of the given edge for each sample, in nats per image.

        Gradients reach phi through ``kernel`` only; x_t and eps_theta are constants.
        """
        t = self.edge_t.to(edges.device)[edges]
        s = self.edge_s.to(edges.device)[edges]
        alpha_t = self.schedule.alpha_bar_at(t, x0)
        alpha_s = self.schedule.alpha_bar_at(s, x0)
        x_t = forward_sample(x0, alpha_t, noise)
        transition = kernel(x_t, epsilon_fn(x_t, t, labels), t, s, labels)
        target_mean, target_variance = forward_posterior(x0, x_t, alpha_t, alpha_s)
        kl = gaussian_kl(target_mean, target_variance, transition.mean, transition.variance)
        return {'kl': kl, 't': t,
                'multiplier': (transition.variance / target_variance).detach(),
                'requested_multiplier': (transition.requested / target_variance).detach(),
                'capped': transition.capped.detach()}

    def __call__(self, epsilon_fn: EpsilonFn, kernel: nn.Module, x0: torch.Tensor,
                 labels: torch.Tensor, *, generator: torch.Generator) -> tuple[torch.Tensor, dict[str, Any]]:
        """Unbiased single-edge estimate of sum_k KL_k, in nats per dimension."""
        edges = self.stratified_edges(len(x0), generator)
        noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        details = self.edge_kl(epsilon_fn, kernel, x0, labels, edges, noise)
        per_sample = self.num_edges * details['kl'] / x0[0].numel()
        return per_sample.mean(), {**details, 'per_sample': per_sample.detach()}

    @torch.no_grad()
    def boundary_terms(self, epsilon_fn: EpsilonFn, kernel: nn.Module, x0: torch.Tensor,
                       labels: torch.Tensor, noise: torch.Tensor) -> dict[str, torch.Tensor]:
        """phi-independent recon and prior terms, in nats per image.

        The decoder is a discretized Gaussian centred on the terminal edge's
        mean (the x0_hat the sampler returns) with the forward variance
        1 - alpha_bar[t_1], the fixedlarge convention for that edge.
        """
        first = torch.full_like(labels, self.grid[-2])
        clean = torch.full_like(labels, CLEAN)
        alpha_first = self.schedule.alpha_bar_at(first, x0)
        x_first = forward_sample(x0, alpha_first, noise)
        terminal = kernel(x_first, epsilon_fn(x_first, first, labels), first, clean, labels)
        recon = -discretized_gaussian_log_likelihood(x0, terminal.mean, (1.0 - alpha_first).expand_as(x0))
        alpha_top = self.schedule.alpha_bar_at(torch.full_like(labels, self.grid[0]), x0)
        prior = gaussian_kl(alpha_top.sqrt() * x0, (1.0 - alpha_top).expand_as(x0),
                            torch.zeros_like(x0), torch.ones_like(x0))
        return {'recon': recon, 'prior': prior}


    def edge_diagnostics(self, epsilon_fn: EpsilonFn, kernel: nn.Module, x0: torch.Tensor,
                         labels: torch.Tensor, *, generator: torch.Generator) -> dict[str, Any]:
        """Every edge of the sampler evaluated on the whole batch: where the bound and its gradient live.

        One row per edge, terminal edge included. ``edge_weight`` is the weight
        of the edge in the trained loss (1 for stochastic edges, 0 for the
        terminal reconstruction, which is reported but constant in phi).
        ``fixed_lambda_kl_nats_per_dim`` re-scores the same states with a
        constant multiplier, which shows where each edge's optimum lies
        independently of the current scaffold.
        """
        dimensions = x0[0].numel()
        parameters = [parameter for parameter in kernel.parameters() if parameter.requires_grad]
        rows = []
        for index in range(self.num_edges):
            edges = torch.full((len(x0),), index, dtype=torch.long, device=x0.device)
            t, s = self.edge_t.to(x0.device)[edges], self.edge_s.to(x0.device)[edges]
            alpha_t, alpha_s = self.schedule.alpha_bar_at(t, x0), self.schedule.alpha_bar_at(s, x0)
            noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
            x_t = forward_sample(x0, alpha_t, noise)
            with torch.no_grad():
                epsilon = epsilon_fn(x_t, t, labels)
            target_mean, target_variance = forward_posterior(x0, x_t, alpha_t, alpha_s)
            with torch.enable_grad():
                transition = kernel(x_t, epsilon, t, s, labels)
                value = gaussian_kl(target_mean, target_variance, transition.mean, transition.variance).mean() / dimensions
                gradients = torch.autograd.grad(value, parameters, allow_unused=True) if parameters else ()
            norm = torch.sqrt(sum((gradient.square().sum() for gradient in gradients if gradient is not None),
                                  start=x0.new_zeros(())))
            with torch.no_grad():
                sweep = {}
                for lam in DIAGNOSTIC_LAMBDAS:
                    fixed = scaffold_transition(x_t, epsilon, lam, alpha_t, alpha_s)
                    sweep[str(lam)] = float(gaussian_kl(target_mean, target_variance, fixed.mean, fixed.variance).mean()) / dimensions
            budget = float(((1.0 - alpha_s) / target_variance).flatten()[0])
            rows.append({
                'edge': [int(self.edge_t[index]), int(self.edge_s[index])], 'is_terminal': False, 'edge_weight': 1.0,
                'edge_kl_nats_per_dim': float(value.detach()), 'edge_gradient_norm': float(norm),
                'lambda_requested_mean': float((transition.requested / target_variance).mean().detach()),
                'lambda_effective_mean': float((transition.variance / target_variance).mean().detach()),
                'variance_clip_fraction': float(transition.capped.float().mean()),
                'lambda_budget': budget,
                'fixed_lambda_kl_nats_per_dim': sweep, 'best_fixed_lambda': float(min(sweep, key=sweep.get)),
            })
        noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        boundary = self.boundary_terms(epsilon_fn, kernel, x0, labels, noise)
        recon, prior = float(boundary['recon'].mean()) / dimensions, float(boundary['prior'].mean()) / dimensions
        rows.append({'edge': [self.grid[-2], CLEAN], 'is_terminal': True, 'edge_weight': 0.0,
                     'edge_kl_nats_per_dim': recon, 'edge_gradient_norm': 0.0,
                     'lambda_requested_mean': None, 'lambda_effective_mean': 0.0, 'variance_clip_fraction': 0.0,
                     'lambda_budget': 0.0, 'fixed_lambda_kl_nats_per_dim': {}, 'best_fixed_lambda': None})
        stochastic = rows[:-1]
        trained = sum(row['edge_kl_nats_per_dim'] for row in stochastic)
        reachable = sum(row['edge_kl_nats_per_dim'] for row in stochastic if row['edge_gradient_norm'] > 0)
        dominant = max(rows, key=lambda row: row['edge_kl_nats_per_dim'])
        return {'edges': rows, 'summary': {
            'full_bound_nats_per_dim': prior + trained + recon, 'prior_nats_per_dim': prior,
            'reconstruction_nats_per_dim': recon, 'trained_loss_nats_per_dim': trained,
            'trained_loss_with_gradient_nats_per_dim': reachable,
            'dominant_edge': dominant['edge'],
            'dominant_edge_share_of_full_bound': dominant['edge_kl_nats_per_dim'] / (prior + trained + recon),
            'zero_gradient_edges': [row['edge'] for row in stochastic if row['edge_gradient_norm'] == 0],
            'edges_preferring_lambda_below_one': [row['edge'] for row in stochastic if row['best_fixed_lambda'] < 1],
            'caveat': self.CAVEAT}}


@register_objective('respaced_vlb')
def _respaced_vlb(settings: dict[str, Any], schedule: Schedule, grid: Grid) -> RespacedVLB:
    return RespacedVLB(schedule, grid)


def build_loss(settings: dict[str, Any], schedule: Schedule, grid: Grid) -> Objective:
    """Objective named by ``loss.kind``; ``register_objective`` adds new kinds."""
    kind = settings['kind']
    if kind not in OBJECTIVES:
        raise ValueError(f'loss.kind must be one of {sorted(OBJECTIVES)}, got {kind!r}')
    return OBJECTIVES[kind](settings, schedule, grid)
