"""The respaced bound scores the sampler's own transitions and trains only phi."""
from __future__ import annotations

import pytest
import torch

from scaffold.backbone import FixedScaffold, MLPScaffold, TimeOnlyScaffold
from scaffold.loss import (
    RespacedVLB, discretized_gaussian_log_likelihood, forward_posterior, forward_sample, gaussian_kl,
)
from scaffold.sampling import CLEAN, ScaffoldKernel, Schedule, respaced_grid, sample, stride_grid


@pytest.fixture()
def problem():
    schedule = Schedule.linear(1000, 1e-4, 0.02)
    generator = torch.Generator().manual_seed(0)
    x0 = torch.rand(16, 3, 8, 8, generator=generator) * 2 - 1
    labels = torch.arange(16) % 4
    return schedule, RespacedVLB(schedule, respaced_grid(1000, 8)), x0, labels, generator


def test_bound_requires_clean_terminal_grid(problem):
    schedule, *_ = problem
    with pytest.raises(ValueError):
        RespacedVLB(schedule, stride_grid(1000, 100))
    with pytest.raises(ValueError):
        RespacedVLB(schedule, (500, CLEAN))


def test_kl_vanishes_for_lambda1_with_the_true_noise(problem):
    """With eps_theta equal to the drawn noise, x0_hat = x0 and lambda=1 is the exact posterior."""
    schedule, loss, x0, labels, generator = problem
    noise = torch.randn(x0.shape, generator=generator)
    edges = loss.stratified_edges(len(x0), generator)
    details = loss.edge_kl(lambda x, t, y: noise, ScaffoldKernel(schedule, FixedScaffold(1.0)), x0, labels, edges, noise)
    assert float(details['kl'].abs().max()) < 1e-2
    assert torch.allclose(details['multiplier'], torch.ones_like(details['multiplier']), atol=1e-5)
    off = loss.edge_kl(lambda x, t, y: noise, ScaffoldKernel(schedule, FixedScaffold(0.5)), x0, labels, edges, noise)
    assert bool((off['kl'] > details['kl'] + 1.0).all())


def test_kl_matches_monte_carlo_of_the_sampling_kernel(problem):
    """The density scored by the loss is the one ``sample`` draws from on that edge."""
    schedule, loss, x0, labels, generator = problem
    kernel = ScaffoldKernel(schedule, FixedScaffold(0.7))
    epsilon = lambda x, t, y: 0.3 * x
    t, s = loss.grid[2], loss.grid[3]
    x_t = forward_sample(x0[:1], schedule.alpha_bar_at(torch.tensor([t]), x0[:1]), torch.randn(x0[:1].shape, generator=generator))
    draws = sample(epsilon, kernel, torch.zeros(20_000, dtype=torch.long), x_t.expand(20_000, -1, -1, -1).clone(),
                   (t, s), generator=generator)
    transition = kernel(x_t, epsilon(x_t, None, None), torch.tensor([t]), torch.tensor([s]), labels[:1])
    assert torch.allclose(draws.mean(0), transition.mean[0], atol=0.02)
    assert torch.allclose(draws.var(0), transition.variance[0], rtol=0.06)


def test_forward_posterior_is_consistent_with_the_forward_process(problem):
    """E[x_s | x_t, x0] and its variance agree with samples of the forward chain."""
    schedule, _, x0, _, generator = problem
    x0 = x0[:1, :1, :1, :1].double().expand(200_000, 1, 1, 1).contiguous()
    t, s = torch.full((len(x0),), 400), torch.full((len(x0),), 300)
    alpha_t, alpha_s = schedule.alpha_bar_at(t, x0), schedule.alpha_bar_at(s, x0)
    x_s = forward_sample(x0, alpha_s, torch.randn(x0.shape, dtype=torch.float64, generator=generator))
    step = alpha_t / alpha_s
    x_t = step.sqrt() * x_s + (1 - step).sqrt() * torch.randn(x0.shape, dtype=torch.float64, generator=generator)
    mean, variance = forward_posterior(x0, x_t, alpha_t, alpha_s)
    residual = x_s - mean
    assert float(residual.mean().abs()) < 2e-3
    assert torch.allclose(residual.var(), variance.flatten()[0], rtol=0.02)


def test_stratified_edges_cover_every_edge_equally(problem):
    _, loss, *_ , generator = problem
    edges = loss.stratified_edges(loss.num_edges * 5, generator)
    assert torch.equal(torch.bincount(edges, minlength=loss.num_edges), torch.full((loss.num_edges,), 5))


@pytest.mark.parametrize('time_only', [False, True])
def test_gradient_reaches_phi_only_through_the_kernel(problem, time_only):
    schedule, loss, x0, labels, generator = problem
    scaffold = MLPScaffold(4, image_size=8, hidden_dims=(16, 16), condition_dim=8)
    torch.nn.init.normal_(scaffold.output.weight, std=0.01)
    module = TimeOnlyScaffold(scaffold) if time_only else scaffold
    value, details = loss(lambda x, t, y: 0.2 * x, ScaffoldKernel(schedule, module), x0, labels, generator=generator)
    value.backward()
    gradients = [p.grad for p in scaffold.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    assert any(float(g.abs().sum()) > 0 for g in gradients)
    assert details['per_sample'].shape == (len(x0),) and bool(torch.isfinite(value))


def test_gaussian_kl_closed_form():
    mean_q, mean_p = torch.zeros(1, 2), torch.tensor([[1.0, 0.0]])
    value = gaussian_kl(mean_q, torch.full((1, 2), 2.0), mean_p, torch.ones(1, 2))
    expected = 2 * 0.5 * (2 - 1 - torch.log(torch.tensor(2.0))) + 0.5
    assert torch.allclose(value, expected.reshape(1))
    with pytest.raises(ValueError):
        gaussian_kl(mean_q, torch.ones(1, 2), mean_p, torch.zeros(1, 2))


def test_discretized_decoder_is_a_normalised_distribution():
    levels = (torch.arange(256, dtype=torch.float64) / 127.5 - 1).reshape(256, 1)
    for mean, variance in ((0.1, 0.03), (-0.98, 0.001), (0.7, 0.5)):
        log_probability = discretized_gaussian_log_likelihood(
            levels, torch.full_like(levels, mean), torch.full_like(levels, variance))
        assert abs(float(log_probability.exp().sum()) - 1.0) < 1e-6


def test_boundary_terms_do_not_depend_on_phi(problem):
    """The terminal edge is deterministic, so recon/prior are constants of the scaffold."""
    schedule, loss, x0, labels, generator = problem
    noise = torch.randn(x0.shape, generator=generator)
    epsilon = lambda x, t, y: 0.2 * x
    a = loss.boundary_terms(epsilon, ScaffoldKernel(schedule, FixedScaffold(0.3)), x0, labels, noise)
    b = loss.boundary_terms(epsilon, ScaffoldKernel(schedule, FixedScaffold(1.9)), x0, labels, noise)
    assert torch.equal(a['recon'], b['recon']) and torch.equal(a['prior'], b['prior'])
    assert bool(torch.isfinite(a['recon']).all()) and bool((a['prior'] >= 0).all())
