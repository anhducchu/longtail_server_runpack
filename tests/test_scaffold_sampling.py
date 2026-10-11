"""Gate A (baseline parity with the vendor samplers) and Gate B (numerical correctness)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from scaffold.backbone import FixedScaffold
from scaffold.sampling import (
    BUDGET_MARGIN, CLEAN, DDPMKernel, ScaffoldKernel, Schedule, build_grid, ddpm_transition, posterior_variance,
    respaced_grid, sample, scaffold_transition, stride_grid,
)

VENDOR = Path(__file__).resolve().parents[1] / 'third_party' / 'CCUA-DDPM'
BETA_1, BETA_T, STEPS = 1e-4, 0.02, 1000


@pytest.fixture(scope='module')
def vendor():
    # The vendor file does `from clr import ...`; expose its directory only while importing.
    sys.path.insert(0, str(VENDOR))
    try:
        return importlib.import_module('diffusion')
    finally:
        sys.path.remove(str(VENDOR))
        sys.modules.pop('diffusion', None)


class TinyEpsilon(nn.Module):
    """Stub denoiser with the vendor UNet call signature.

    It is the exact noise prediction for N(0, 0.25 I) data plus a small
    nonlinear, label-dependent term, so chains stay O(1) like a trained model's.
    """

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(3, 3, 3, padding=1)
        self.label = nn.Embedding(10, 3)
        self.register_buffer('alpha_bar', Schedule.linear(STEPS, BETA_1, BETA_T).alpha_bar.float())

    def forward(self, x, t, y=None, return_mid=False):
        alpha = self.alpha_bar[t][:, None, None, None]
        out = (1 - alpha).sqrt() * x / (0.25 * alpha + 1 - alpha)
        residual = torch.tanh(self.conv(x))
        if y is not None:
            residual = residual + self.label(y)[:, :, None, None]
        out = out + 0.1 * alpha.sqrt() * residual
        return (out, out) if return_mid else out


@pytest.fixture(scope='module')
def setup():
    model = TinyEpsilon().requires_grad_(False).eval()
    labels = torch.arange(4) % 10
    initial = torch.randn(4, 3, 8, 8, generator=torch.Generator().manual_seed(1))
    return model, labels, initial, Schedule.linear(STEPS, BETA_1, BETA_T)


def test_schedule_matches_vendor_buffers(vendor, setup):
    model, *_, schedule = setup
    reference = vendor.GaussianDiffusionSampler(model, BETA_1, BETA_T, STEPS, num_class=10)
    assert torch.equal(schedule.alpha_bar, reference.alphas_bar)


def test_grids():
    assert respaced_grid(1000, 20, spacing='trailing') == tuple(range(999, 0, -50)) + (CLEAN,)
    assert respaced_grid(1000, 20, spacing='leading') == tuple(range(950, -1, -50)) + (CLEAN,)
    for spacing in ('leading', 'trailing'):
        assert respaced_grid(1000, 1000, spacing=spacing) == tuple(range(999, -2, -1))
    assert stride_grid(1000, 10) == tuple(range(990, -1, -10))
    assert len(respaced_grid(1000, 50)) - 1 == 50
    assert build_grid(1000, {'stride': 10}) == stride_grid(1000, 10)
    assert build_grid(1000, {'nfe': 20, 'spacing': 'trailing'}) == respaced_grid(1000, 20, spacing='trailing')
    with pytest.raises(ValueError):
        build_grid(1000, {'stride': 10, 'nfe': 20})


def test_lambda0_matches_vendor_ddim(vendor, setup):
    """Gate A: lambda=0 on the vendor stride grid reproduces the vendor DDIM sampler."""
    model, labels, initial, schedule = setup
    reference = vendor.GaussianDiffusionSamplerDDIM(model, BETA_1, BETA_T, STEPS, omega=0.0, cond=True)
    with torch.no_grad():
        expected = reference(initial, labels, method='ddim', skip=10)
    ours = sample(lambda x, t, y: model(x, t, y), ScaffoldKernel(schedule, FixedScaffold(0.0)), labels,
                  initial, stride_grid(STEPS, 10), generator=torch.Generator().manual_seed(2))
    assert torch.allclose(ours.clamp(-1, 1), expected, atol=2e-4)


@pytest.mark.parametrize('variance_type', ['fixedlarge', 'fixedsmall'])
def test_ddpm_kernel_matches_vendor_ancestral(vendor, setup, variance_type):
    """Gate A: full-chain ancestral DDPM equals the vendor mean/variance with shared noise."""
    model, labels, initial, schedule = setup
    reference = vendor.GaussianDiffusionSampler(model, BETA_1, BETA_T, STEPS, num_class=10, var_type=variance_type)
    generator = torch.Generator().manual_seed(3)
    expected = initial.clone()
    with torch.no_grad():
        for step in reversed(range(STEPS)):
            mean, log_variance = reference.p_mean_variance(expected, torch.full_like(labels, step), y=labels)
            noise = torch.randn(expected.shape, generator=generator) if step > 0 else 0
            expected = mean + torch.exp(0.5 * log_variance) * noise
    ours = sample(lambda x, t, y: model(x, t, y), DDPMKernel(schedule, variance_type=variance_type), labels,
                  initial, respaced_grid(STEPS, STEPS), generator=torch.Generator().manual_seed(3))
    assert torch.allclose(ours, expected, atol=2e-4)


def test_lambda1_is_fixedsmall_and_stochastic_ddim(setup):
    """lambda=1 equals ancestral fixedsmall and Song et al. eta=1 on a respaced edge."""
    *_, schedule = setup
    x = torch.randn(5, 3, 4, 4, dtype=torch.float64)
    eps = torch.randn_like(x)
    t, s = torch.full((5,), 600), torch.full((5,), 550)
    alpha_t, alpha_s = schedule.alpha_bar_at(t, x), schedule.alpha_bar_at(s, x)
    ours = scaffold_transition(x, eps, 1.0, alpha_t, alpha_s)
    small = ddpm_transition(x, eps, alpha_t, alpha_s, variance_type='fixedsmall')
    assert torch.equal(ours.mean, small.mean) and torch.allclose(ours.variance, small.variance)
    sigma = ((1 - alpha_s) / (1 - alpha_t)).sqrt() * (1 - alpha_t / alpha_s).sqrt()
    x0 = (x - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
    expected_mean = alpha_s.sqrt() * x0 + (1 - alpha_s - sigma ** 2).sqrt() * eps
    assert torch.allclose(ours.mean, expected_mean) and torch.allclose(ours.variance, (sigma ** 2).expand_as(x))
    # The DDIM-form mean at lambda=1 is the DDPM posterior mean evaluated at x0_hat.
    step = alpha_t / alpha_s
    posterior_mean = (alpha_s.sqrt() * (1 - step) * x0 + step.sqrt() * (1 - alpha_s) * x) / (1 - alpha_t)
    assert torch.allclose(ours.mean, posterior_mean)


def test_terminal_edge_returns_x0_hat_without_noise(setup):
    *_, schedule = setup
    x = torch.randn(3, 3, 4, 4, dtype=torch.float64)
    eps = torch.randn_like(x)
    t, clean = torch.full((3,), 49), torch.full((3,), CLEAN)
    transition = ScaffoldKernel(schedule, FixedScaffold(5.0))(x, eps, t, clean, torch.zeros(3, dtype=torch.long))
    assert torch.equal(transition.variance, torch.zeros_like(x))
    assert torch.allclose(transition.mean, transition.x0_hat)

    generator = torch.Generator().manual_seed(0)
    before = generator.get_state().clone()
    out = sample(lambda x_, t_, y_: eps, ScaffoldKernel(schedule, FixedScaffold(5.0)),
                 torch.zeros(3, dtype=torch.long), x, (49, CLEAN), generator=generator)
    assert torch.allclose(out, transition.x0_hat)
    assert torch.equal(generator.get_state(), before), 'terminal edge must not consume noise'


def test_variance_budget_is_enforced(setup):
    *_, schedule = setup
    x = torch.randn(2, 3, 4, 4, dtype=torch.float64)
    t, s = torch.full((2,), 999), torch.full((2,), 949)
    alpha_t, alpha_s = schedule.alpha_bar_at(t, x), schedule.alpha_bar_at(s, x)
    wide = scaffold_transition(x, torch.randn_like(x), 100.0, alpha_t, alpha_s)
    assert bool(wide.capped.all())
    assert torch.allclose(wide.variance, ((1 - BUDGET_MARGIN) * (1 - alpha_s)).expand_as(x))
    assert bool(torch.isfinite(wide.mean).all())
    unit = scaffold_transition(x, torch.randn_like(x), 1.0, alpha_t, alpha_s)
    assert not bool(unit.capped.any())
    assert torch.allclose(unit.variance, posterior_variance(alpha_t, alpha_s).expand_as(x))
    with pytest.raises(ValueError):
        scaffold_transition(x, x, -0.1, alpha_t, alpha_s)


def test_paired_arms_share_brownian_draws(setup):
    """A deterministic arm still advances the stream, so stochastic arms stay aligned."""
    model, labels, initial, schedule = setup
    grid = respaced_grid(STEPS, 10)
    epsilon = lambda x, t, y: model(x, t, y)
    a = sample(epsilon, ScaffoldKernel(schedule, FixedScaffold(1.0)), labels, initial, grid,
               generator=torch.Generator().manual_seed(7))
    b = sample(epsilon, ScaffoldKernel(schedule, FixedScaffold(1.0)), labels, initial, grid,
               generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    used, unused = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    sample(epsilon, ScaffoldKernel(schedule, FixedScaffold(0.0)), labels, initial, grid, generator=used)
    sample(epsilon, ScaffoldKernel(schedule, FixedScaffold(1.0)), labels, initial, grid, generator=unused)
    assert torch.equal(used.get_state(), unused.get_state())


# Gate B: exact-score Gaussian control. Data x0 ~ N(0, diag(std^2)); E[eps | x_t] is closed form,
# so the endpoint law of a fixed-lambda sampler is Gaussian with a variance we can derive by hand.

DATA_STD = torch.tensor([0.5, 0.2, 0.05], dtype=torch.float64)


def analytic_endpoint_ratio(schedule: Schedule, grid: tuple[int, ...], lam: float) -> torch.Tensor:
    """Exact endpoint std / data std of the lambda-kernel under the exact score.

    Each edge is linear, x_s = c x_t + sigma z, so var_s = c^2 var_t + sigma^2 with
    c = (sqrt(ab_s ab_t) std^2 + sqrt((1-ab_s-sigma^2)(1-ab_t))) / (ab_t std^2 + 1-ab_t).
    """
    padded = torch.cat((schedule.alpha_bar, torch.ones(1, dtype=torch.float64)))
    variance = torch.ones(3, dtype=torch.float64)
    for t, s in zip(grid, grid[1:]):
        alpha_t, alpha_s = padded[t], padded[s]
        sigma2 = min(lam * (1 - alpha_s) / (1 - alpha_t) * (1 - alpha_t / alpha_s), (1 - BUDGET_MARGIN) * (1 - alpha_s))
        marginal = alpha_t * DATA_STD ** 2 + 1 - alpha_t
        gain = ((alpha_s * alpha_t).sqrt() * DATA_STD ** 2 + ((1 - alpha_s - sigma2) * (1 - alpha_t)).sqrt()) / marginal
        variance = gain ** 2 * variance + sigma2
    return variance.sqrt() / DATA_STD


def empirical_endpoint_ratio(schedule: Schedule, grid: tuple[int, ...], lam: float, particles: int = 100_000) -> torch.Tensor:
    def epsilon(x, t, labels):
        alpha = schedule.alpha_bar_at(t, x)
        return (1 - alpha).sqrt() * x / (alpha * DATA_STD ** 2 + 1 - alpha)
    generator = torch.Generator().manual_seed(11)
    initial = torch.randn(particles, 3, dtype=torch.float64, generator=generator)
    out = sample(epsilon, ScaffoldKernel(schedule, FixedScaffold(lam)), torch.zeros(particles, dtype=torch.long),
                 initial, grid, generator=generator)
    return out.std(dim=0) / DATA_STD


@pytest.mark.parametrize('lam', [0.0, 1.0])
@pytest.mark.parametrize('spacing', ['leading', 'trailing'])
def test_gaussian_exact_score_error_matches_theory(setup, lam, spacing):
    """The sampler's finite-step error at 20/50/100 NFE is exactly the method's, nothing more."""
    *_, schedule = setup
    for nfe in (20, 50, 100):
        grid = respaced_grid(STEPS, nfe, spacing=spacing)
        expected = analytic_endpoint_ratio(schedule, grid, lam)
        assert torch.allclose(empirical_endpoint_ratio(schedule, grid, lam), expected, rtol=0.02), (nfe, expected)


@pytest.mark.parametrize('lam', [0.0, 1.0])
def test_gaussian_exact_score_error_converges(setup, lam):
    """Under-dispersion is a finite-step effect: it shrinks monotonically with NFE."""
    *_, schedule = setup
    errors = [(analytic_endpoint_ratio(schedule, respaced_grid(STEPS, nfe, spacing='leading'), lam) - 1).abs()
              for nfe in (20, 50, 100, 500)]
    for coarse, fine in zip(errors, errors[1:]):
        assert bool((fine < coarse).all())
    assert float(errors[-1][0]) < 0.02


def test_gaussian_point_mass_is_recovered_exactly(setup):
    """With zero data variance every kernel must land on the data point: no terminal noise."""
    *_, schedule = setup
    point = torch.tensor([0.3, -0.7, 0.1], dtype=torch.float64)

    def epsilon(x, t, labels):
        alpha = schedule.alpha_bar_at(t, x)
        return (x - alpha.sqrt() * point) / (1 - alpha).sqrt()
    for kernel in (ScaffoldKernel(schedule, FixedScaffold(0.0)), ScaffoldKernel(schedule, FixedScaffold(1.7)),
                   DDPMKernel(schedule, variance_type='fixedlarge')):
        generator = torch.Generator().manual_seed(5)
        initial = torch.randn(64, 3, dtype=torch.float64, generator=generator)
        out = sample(epsilon, kernel, torch.zeros(64, dtype=torch.long), initial,
                     respaced_grid(len(schedule), 20), generator=generator)
        assert torch.allclose(out, point.expand_as(out), atol=1e-9)
