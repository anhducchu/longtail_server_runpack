"""sampling.py against numpy oracles: schedule, grids, edges, kernels and the sampling loop."""
import numpy as np
import pytest
import torch

from indep_helpers import CLEAN, ab_at, np_alpha_bar, np_beta_tilde, np_edge, np_posterior_bayes
from scaffold import sampling as S
from scaffold.backbone import FixedScaffold

T = 1000
AB = np_alpha_bar()


@pytest.fixture(scope='module')
def schedule():
    return S.Schedule.linear(T, 1e-4, 0.02)


def column(values, dtype=torch.float64):
    return torch.as_tensor(np.asarray(values), dtype=dtype).reshape(-1, 1, 1, 1)


# Schedule

def test_linear_schedule_matches_numpy(schedule):
    assert schedule.alpha_bar.dtype == torch.float64 and len(schedule) == T
    np.testing.assert_allclose(schedule.alpha_bar.numpy(), AB, rtol=1e-5)
    assert schedule.alpha_bar[0].item() == pytest.approx(0.9999, rel=1e-6)
    assert 3.5e-5 < schedule.alpha_bar[-1].item() < 4.5e-5


@pytest.mark.parametrize('alpha', [
    torch.tensor([0.9, 0.5], dtype=torch.float32),
    torch.tensor([0.9], dtype=torch.float64),
    torch.tensor([0.5, 0.9], dtype=torch.float64),
    torch.tensor([0.9, 0.9], dtype=torch.float64),
    torch.tensor([1.0, 0.5], dtype=torch.float64),
    torch.tensor([0.5, 0.0], dtype=torch.float64),
    torch.tensor([[0.9, 0.5]], dtype=torch.float64),
])
def test_schedule_rejects_invalid_alpha_bar(alpha):
    with pytest.raises(ValueError):
        S.Schedule(alpha)


def test_alpha_bar_at_broadcasts_and_maps_clean_to_one(schedule):
    like = torch.zeros(4, 3, 8, 8)
    values = schedule.alpha_bar_at(torch.tensor([0, 500, 999, CLEAN]), like)
    assert values.shape == (4, 1, 1, 1) and values.dtype == like.dtype
    np.testing.assert_allclose(values.flatten().numpy(), [AB[0], AB[500], AB[999], 1.0], rtol=1e-5)
    assert values[3].item() == 1.0


@pytest.mark.parametrize('indices', [
    torch.tensor([0.0, 1.0]), torch.tensor([0, 1, 2]), torch.tensor([[0, 1]]),
    torch.tensor([0, -2]), torch.tensor([0, T]),
])
def test_alpha_bar_at_rejects_bad_indices(schedule, indices):
    with pytest.raises(ValueError):
        schedule.alpha_bar_at(indices, torch.zeros(2, 3))


def test_time_is_index_plus_one_over_T(schedule):
    np.testing.assert_allclose(schedule.time(torch.tensor([0, 499, 999])).numpy(), [1e-3, 0.5, 1.0], rtol=1e-6)


# Grids

def test_documented_grids():
    assert S.respaced_grid(1000, 20, spacing='leading') == tuple(range(950, -1, -50)) + (CLEAN,)
    assert S.respaced_grid(1000, 20, spacing='trailing') == tuple(range(999, 48, -50)) + (CLEAN,)
    full = tuple(range(999, -1, -1)) + (CLEAN,)
    assert S.respaced_grid(1000, 1000, spacing='leading') == full
    assert S.respaced_grid(1000, 1000, spacing='trailing') == full
    assert S.stride_grid(1000, 10) == tuple(range(990, -1, -10))
    assert S.build_grid(1000, {'stride': 10}) == S.stride_grid(1000, 10)
    assert S.build_grid(1000, {'nfe': 20, 'spacing': 'leading'}) == S.respaced_grid(1000, 20)


@pytest.mark.parametrize('spacing', ['leading', 'trailing'])
@pytest.mark.parametrize('steps', [1000, 37])
def test_every_nfe_gives_a_valid_K_edge_grid(spacing, steps):
    for nfe in range(1, steps + 1):
        grid = S.respaced_grid(steps, nfe, spacing=spacing)
        assert len(grid) == nfe + 1 and grid[-1] == CLEAN, (nfe, grid)
        assert all(a > b for a, b in zip(grid, grid[1:])), (nfe, grid)
        assert 0 <= grid[-2] and grid[0] <= steps - 1, (nfe, grid)
        if spacing == 'trailing':
            assert grid[0] == steps - 1
        else:
            assert grid[-2] == 0


@pytest.mark.parametrize('nfe', [0, 1001, True, 20.0, -3])
def test_respaced_grid_rejects_bad_nfe(nfe):
    with pytest.raises(ValueError):
        S.respaced_grid(1000, nfe)


def test_grid_validation_errors():
    with pytest.raises(ValueError):
        S.respaced_grid(1000, 20, spacing='uniform')
    for stride in (0, 7, 1000):
        with pytest.raises(ValueError):
            S.stride_grid(1000, stride)
    with pytest.raises(ValueError):
        S.build_grid(1000, {'stride': 10, 'nfe': 20})
    for grid in ((5,), (5, 5), (3, 5), (1000, 5), (5, -2)):
        with pytest.raises(ValueError):
            S.validate_grid(grid, 1000)


# Single edges

def random_edge(rng, batch=6, shape=(3, 4, 4)):
    t = rng.integers(1, T, size=batch)
    s = np.array([rng.integers(0, value) for value in t])
    return (rng.normal(size=(batch, *shape)), rng.normal(size=(batch, *shape)), ab_at(AB, t), ab_at(AB, s))


@pytest.mark.parametrize('lam', [0.0, 0.25, 1.0, 1.5, 50.0])
def test_scaffold_transition_matches_readme_formula(lam):
    x, eps, ab_t, ab_s = random_edge(np.random.default_rng(int(lam * 100)))
    x0_hat, mean, variance, capped = np_edge(x, eps, lam, ab_t, ab_s)
    got = S.scaffold_transition(torch.from_numpy(x), torch.from_numpy(eps), lam, column(ab_t), column(ab_s))
    np.testing.assert_allclose(got.x0_hat.numpy(), x0_hat, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(got.mean.numpy(), mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(got.variance.numpy(), np.broadcast_to(variance, x.shape), rtol=1e-10, atol=0)
    assert got.variance.shape == x.shape and got.capped.shape == x.shape
    np.testing.assert_array_equal(got.capped.numpy(), np.broadcast_to(capped, x.shape))
    assert torch.isfinite(got.mean).all()


def test_per_coordinate_lambda_is_applied_per_coordinate():
    rng = np.random.default_rng(1)
    x, eps, ab_t, ab_s = random_edge(rng)
    lam = rng.uniform(0, 3, size=x.shape)
    _, mean, variance, capped = np_edge(x, eps, lam, ab_t, ab_s)
    got = S.scaffold_transition(torch.from_numpy(x), torch.from_numpy(eps), torch.from_numpy(lam),
                                column(ab_t), column(ab_s))
    np.testing.assert_allclose(got.mean.numpy(), mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(got.variance.numpy(), variance, rtol=1e-10)
    np.testing.assert_array_equal(got.capped.numpy(), capped)


@pytest.mark.parametrize('lam', [0.0, 0.3, 1.0, 1.7, 1e6])
def test_every_lambda_preserves_the_marginal_under_the_true_noise(lam):
    """mean = sqrt(ab_s) x0 + c eps with c^2 + sigma^2 = 1 - ab_s, so x_s ~ q(x_s | x0) whatever lambda is."""
    rng = np.random.default_rng(2)
    x0, eps, ab_t, ab_s = random_edge(rng)
    x_t = np.sqrt(ab_t) * x0 + np.sqrt(1 - ab_t) * eps
    got = S.scaffold_transition(torch.from_numpy(x_t), torch.from_numpy(eps), lam, column(ab_t), column(ab_s))
    np.testing.assert_allclose(got.x0_hat.numpy(), x0, atol=1e-9)
    coefficient = (got.mean.numpy() - np.sqrt(ab_s) * x0) / eps
    np.testing.assert_allclose(coefficient ** 2 + got.variance.numpy(), np.broadcast_to(1 - ab_s, x0.shape), rtol=1e-8)


def test_lambda_one_is_the_bayes_posterior():
    rng = np.random.default_rng(3)
    x0, eps, ab_t, ab_s = random_edge(rng)
    x_t = np.sqrt(ab_t) * x0 + np.sqrt(1 - ab_t) * eps
    mean, variance = np_posterior_bayes(x0, x_t, ab_t, ab_s)
    got = S.scaffold_transition(torch.from_numpy(x_t), torch.from_numpy(eps), 1.0, column(ab_t), column(ab_s))
    np.testing.assert_allclose(got.mean.numpy(), mean, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(got.variance.numpy(), np.broadcast_to(variance, x0.shape), rtol=1e-8)
    np.testing.assert_allclose(S.posterior_variance(column(ab_t), column(ab_s)).numpy(), variance, rtol=1e-8)


def test_clean_edge_returns_x0_hat_without_noise():
    rng = np.random.default_rng(4)
    x, eps, ab_t, _ = random_edge(rng)
    for lam in (0.0, 1.0, 7.0):
        got = S.scaffold_transition(torch.from_numpy(x), torch.from_numpy(eps), lam, column(ab_t),
                                    column(np.ones(len(x))))
        assert (got.variance == 0).all() and not got.capped.any()
        np.testing.assert_allclose(got.mean.numpy(), (x - np.sqrt(1 - ab_t) * eps) / np.sqrt(ab_t), rtol=1e-12)
        np.testing.assert_array_equal(got.sample(torch.randn_like(got.mean)).numpy(), got.mean.numpy())


def test_budget_caps_variance_and_keeps_the_mean_finite():
    rng = np.random.default_rng(5)
    x, eps, ab_t, ab_s = random_edge(rng)
    got = S.scaffold_transition(torch.from_numpy(x), torch.from_numpy(eps), 1e9, column(ab_t), column(ab_s))
    assert got.capped.all() and torch.isfinite(got.mean).all()
    np.testing.assert_allclose(got.variance.numpy(), np.broadcast_to((1 - 1e-4) * (1 - ab_s), x.shape), rtol=1e-12)


@pytest.mark.parametrize('lam', [-1e-6, float('nan'), float('inf')])
def test_scaffold_transition_rejects_bad_lambda(lam):
    one = torch.ones(1, 1)
    with pytest.raises(ValueError):
        S.scaffold_transition(one, one, lam, one * 0.5, one * 0.8)


def test_ddpm_transition_matches_the_adjacent_step_ddpm_update():
    """For s = t - 1 the kernel must be the textbook DDPM step with beta_t / beta~_t."""
    rng = np.random.default_rng(6)
    betas = np.linspace(1e-4, 0.02, T)
    t = rng.integers(1, T, size=5)
    x, eps = rng.normal(size=(5, 3, 2, 2)), rng.normal(size=(5, 3, 2, 2))
    ab_t, ab_s, beta = ab_at(AB, t), ab_at(AB, t - 1), betas[t].reshape(-1, 1, 1, 1)
    mean = (x - beta / np.sqrt(1 - ab_t) * eps) / np.sqrt(1 - beta)
    for variance_type, variance in (('fixedlarge', beta), ('fixedsmall', beta * (1 - ab_s) / (1 - ab_t))):
        got = S.ddpm_transition(torch.from_numpy(x), torch.from_numpy(eps), column(ab_t), column(ab_s),
                                variance_type=variance_type)
        np.testing.assert_allclose(got.mean.numpy(), mean, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(got.variance.numpy(), np.broadcast_to(variance, x.shape), rtol=1e-7)
        assert not got.capped.any()
    with pytest.raises(ValueError):
        S.ddpm_transition(torch.from_numpy(x), torch.from_numpy(eps), column(ab_t), column(ab_s), variance_type='x')
    with pytest.raises(ValueError):
        S.DDPMKernel(S.Schedule.linear(T, 1e-4, 0.02), variance_type='learned')


def test_float32_edges_track_the_float64_oracle_on_the_production_grids(schedule):
    """Production runs in float32; the worst edge of every grid must stay close to float64 maths."""
    rng = np.random.default_rng(7)
    worst_mean = worst_variance = 0.0
    for grid in (S.respaced_grid(T, 20), S.respaced_grid(T, 20, spacing='trailing'), S.respaced_grid(T, T),
                 S.stride_grid(T, 10)):
        t, s = np.array(grid[:-1]), np.array(grid[1:])
        x, eps = rng.normal(size=(len(t), 3, 4, 4)), rng.normal(size=(len(t), 3, 4, 4))
        for lam in (0.0, 1.0, 2.0):
            _, mean, variance, _ = np_edge(x, eps, lam, ab_at(AB, t), ab_at(AB, s))
            kernel = S.ScaffoldKernel(schedule, FixedScaffold(lam))
            got = kernel(torch.from_numpy(x).float(), torch.from_numpy(eps).float(), torch.from_numpy(t),
                         torch.from_numpy(s), torch.zeros(len(t), dtype=torch.long))
            assert got.mean.dtype == torch.float32
            worst_mean = max(worst_mean, float(np.abs(got.mean.numpy() - mean).max()))
            reference = np.broadcast_to(variance, x.shape)
            positive = reference > 0
            relative = np.abs(got.variance.numpy()[positive] / reference[positive] - 1)
            worst_variance = max(worst_variance, float(relative.max(initial=0.0)))
            assert (got.variance.numpy()[~positive] == 0).all()
    print(f'float32 worst |mean error| = {worst_mean:.3e}, worst relative variance error = {worst_variance:.3e}')
    assert worst_mean < 2e-3 and worst_variance < 5e-3


def test_lambda_up_to_one_is_never_capped_on_the_real_schedule():
    """'ddim_eta1' is only eta = 1 if the budget never binds for lambda <= 1."""
    for grid in (S.respaced_grid(T, n, spacing=sp) for n in (5, 10, 20, 50, 100, 1000) for sp in ('leading', 'trailing')):
        t, s = np.array(grid[:-2]), np.array(grid[1:-1])
        beta_tilde, budget = np_beta_tilde(AB[t], AB[s]), (1 - 1e-4) * (1 - AB[s])
        assert (beta_tilde <= budget).all(), grid


# Kernels and the loop

class Recorder:
    def __init__(self, function):
        self.function, self.timesteps = function, []

    def __call__(self, x, t, labels):
        assert t.shape == labels.shape and not t.is_floating_point()
        self.timesteps.append(int(t[0]))
        return self.function(x, t, labels)


def linear_epsilon(x, t, labels):
    return 0.3 * x + 0.01 * labels.to(x).reshape(-1, 1, 1, 1) + 1e-4 * t.to(x).reshape(-1, 1, 1, 1)


def np_linear_epsilon(x, t, labels):
    return 0.3 * x + 0.01 * labels.reshape(-1, 1, 1, 1) + 1e-4 * t


def np_sample(grid, labels, initial, noises, *, lam=None, ddpm=None):
    """Reference loop: one supplied noise tensor per non-terminal edge."""
    state, noises = initial.copy(), list(noises)
    for t, s in zip(grid, grid[1:]):
        ab_t, ab_s = ab_at(AB, t), ab_at(AB, s)
        eps = np_linear_epsilon(state, t, labels)
        _, mean, variance, _ = np_edge(state, eps, 1.0 if ddpm else lam, ab_t, ab_s)
        if ddpm == 'fixedlarge':
            variance = 1.0 - ab_t / ab_s
        state = mean if s == CLEAN else mean + np.sqrt(variance) * noises.pop(0)
    assert not noises
    return state


def kernels(schedule):
    return {'lam0': (S.ScaffoldKernel(schedule, FixedScaffold(0.0)), {'lam': 0.0}),
            'lam0.5': (S.ScaffoldKernel(schedule, FixedScaffold(0.5)), {'lam': 0.5}),
            'lam2': (S.ScaffoldKernel(schedule, FixedScaffold(2.0)), {'lam': 2.0}),
            'small': (S.DDPMKernel(schedule, variance_type='fixedsmall'), {'ddpm': 'fixedsmall'}),
            'large': (S.DDPMKernel(schedule, variance_type='fixedlarge'), {'ddpm': 'fixedlarge'})}


@pytest.mark.parametrize('name', ['lam0', 'lam0.5', 'lam2', 'small', 'large'])
@pytest.mark.parametrize('grid', [S.respaced_grid(T, 20), S.respaced_grid(T, 7, spacing='trailing'),
                                  S.stride_grid(T, 100)], ids=['leading20', 'trailing7', 'stride100'])
def test_sample_reproduces_an_independent_loop_with_shared_noise(schedule, name, grid):
    kernel, oracle = kernels(schedule)[name]
    labels = torch.tensor([0, 3, 7, 1])
    initial = torch.randn(4, 3, 4, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(11))
    draws = sum(1 for s in grid[1:] if s != CLEAN)
    replay = torch.Generator().manual_seed(99)
    noises = [torch.randn(initial.shape, dtype=torch.float64, generator=replay).numpy() for _ in range(draws)]
    recorder = Recorder(linear_epsilon)
    untouched = initial.clone()
    got = S.sample(recorder, kernel, labels, initial, grid, generator=torch.Generator().manual_seed(99))
    expected = np_sample(grid, labels.numpy(), initial.numpy(), noises, **oracle)
    # The oracle's alpha_bar is pure float64; the repo's starts from float32 betas (relative 1e-7).
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-4, atol=1e-4)
    assert recorder.timesteps == list(grid[:-1]), 'one theta evaluation per edge, at the edge start'
    assert torch.equal(initial, untouched), 'initial_noise must not be modified in place'


def test_deterministic_and_stochastic_arms_consume_the_same_noise(schedule):
    grid, labels = S.respaced_grid(T, 6), torch.tensor([1, 2])
    initial = torch.randn(2, 3, 4, 4, generator=torch.Generator().manual_seed(0))
    states = []
    for kernel, _ in kernels(schedule).values():
        generator = torch.Generator().manual_seed(5)
        S.sample(linear_epsilon, kernel, labels, initial, grid, generator=generator)
        states.append(torch.randn(3, generator=generator))
    assert all(torch.equal(states[0], other) for other in states[1:])
    fresh = torch.Generator().manual_seed(5)
    for _ in range(5):  # 6 edges, the CLEAN edge draws nothing
        torch.randn(initial.shape, generator=fresh)
    assert torch.equal(states[0], torch.randn(3, generator=fresh))


def test_sample_is_reproducible_and_seed_sensitive(schedule):
    kernel = S.ScaffoldKernel(schedule, FixedScaffold(1.0))
    grid, labels = S.respaced_grid(T, 5), torch.tensor([0, 1, 2])
    initial = torch.randn(3, 3, 4, 4, generator=torch.Generator().manual_seed(0))
    run = lambda seed: S.sample(linear_epsilon, kernel, labels, initial, grid,
                                generator=torch.Generator().manual_seed(seed))
    assert torch.equal(run(1), run(1)) and not torch.equal(run(1), run(2))
    ddim = S.ScaffoldKernel(schedule, FixedScaffold(0.0))
    assert torch.equal(S.sample(linear_epsilon, ddim, labels, initial, grid, generator=torch.Generator().manual_seed(1)),
                       S.sample(linear_epsilon, ddim, labels, initial, grid, generator=torch.Generator().manual_seed(2)))


def test_sample_rejects_bad_inputs(schedule):
    kernel = S.ScaffoldKernel(schedule, FixedScaffold(0.0))
    initial, generator = torch.randn(3, 3, 4, 4), torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        S.sample(linear_epsilon, kernel, torch.tensor([0, 1]), initial, S.respaced_grid(T, 5), generator=generator)
    with pytest.raises(ValueError):
        S.sample(linear_epsilon, kernel, torch.tensor([0, 1, 2]), initial, (5, 7, 3), generator=generator)
    with pytest.raises(ValueError):
        S.sample(linear_epsilon, kernel, torch.tensor([0, 1, 2]), initial, (1000, 3), generator=generator)
    with pytest.raises(FloatingPointError):
        S.sample(lambda x, t, y: x * float('nan'), kernel, torch.tensor([0, 1, 2]), initial,
                 S.respaced_grid(T, 5), generator=generator)


# Exact-score Gaussian control: the sampler is linear, so its endpoint law is known in closed form.

MU, SD = np.array([0.3, -0.2, 0.1]), np.array([0.5, 0.2, 0.05])


def gaussian_epsilon(x, t, labels):
    alpha = torch.from_numpy(AB)[t].reshape(-1, 1, 1, 1)
    mu, variance = torch.from_numpy(MU).reshape(1, 3, 1, 1), torch.from_numpy(SD ** 2).reshape(1, 3, 1, 1)
    return (1 - alpha).sqrt() * (x - alpha.sqrt() * mu) / (alpha * variance + 1 - alpha)


def endpoint_law(grid, *, lam=None, ddpm=None):
    """Mean and std of the endpoint for data N(MU, SD^2) under the exact score, by propagating moments."""
    mean, variance = np.zeros(3), np.ones(3)
    for t, s in zip(grid, grid[1:]):
        ab_t, ab_s = AB[t], (1.0 if s == CLEAN else AB[s])
        k = np.sqrt(1 - ab_t) / (ab_t * SD ** 2 + 1 - ab_t)       # eps = k (x - sqrt(ab_t) MU)
        beta_tilde = np_beta_tilde(ab_t, ab_s)
        in_mean = beta_tilde if ddpm else min(lam * beta_tilde, (1 - 1e-4) * (1 - ab_s))
        injected = (1 - ab_t / ab_s) if ddpm == 'fixedlarge' else in_mean
        if s == CLEAN:
            injected = 0.0
        ratio, direction = np.sqrt(ab_s / ab_t), np.sqrt(1 - ab_s - in_mean)
        slope = ratio * (1 - np.sqrt(1 - ab_t) * k) + direction * k
        offset = (ratio * np.sqrt(1 - ab_t) - direction) * k * np.sqrt(ab_t) * MU
        mean, variance = slope * mean + offset, slope ** 2 * variance + injected
    return mean, np.sqrt(variance)


@pytest.mark.parametrize('name,oracle', [('lam0', {'lam': 0.0}), ('lam0.5', {'lam': 0.5}), ('lam2', {'lam': 2.0}),
                                         ('small', {'ddpm': 'fixedsmall'}), ('large', {'ddpm': 'fixedlarge'})])
@pytest.mark.parametrize('grid', [S.respaced_grid(T, 20), S.respaced_grid(T, 20, spacing='trailing'),
                                  S.respaced_grid(T, 50)], ids=['leading20', 'trailing20', 'leading50'])
def test_gaussian_control_endpoint_matches_closed_form(schedule, name, oracle, grid):
    count = 60_000
    initial = torch.randn(count, 3, 1, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(21))
    samples = S.sample(gaussian_epsilon, kernels(schedule)[name][0], torch.zeros(count, dtype=torch.long), initial,
                       grid, generator=torch.Generator().manual_seed(22)).reshape(count, 3).numpy()
    mean, std = endpoint_law(grid, **oracle)
    np.testing.assert_allclose(samples.std(axis=0), std, rtol=0.02)
    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=6 * std.max() / np.sqrt(count))
    np.testing.assert_allclose(mean, MU, atol=0.02)


def test_full_ancestral_chain_recovers_the_data_distribution(schedule):
    """Positive control for the whole loop: 1000 exact-score posterior steps give back N(MU, SD^2).

    The terminal edge returns x0_hat from index 0 (noise std 0.01), which shrinks the
    narrowest direction (std 0.05) by about 7 %; the closed form says exactly how much.
    """
    count, grid = 20_000, S.respaced_grid(T, T)
    initial = torch.randn(count, 3, 1, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(31))
    samples = S.sample(gaussian_epsilon, S.DDPMKernel(schedule, variance_type='fixedsmall'),
                       torch.zeros(count, dtype=torch.long), initial, grid,
                       generator=torch.Generator().manual_seed(32)).reshape(count, 3).numpy()
    np.testing.assert_allclose(samples.mean(axis=0), MU, atol=0.02)
    np.testing.assert_allclose(samples.std(axis=0), endpoint_law(grid, ddpm='fixedsmall')[1], rtol=0.02)
    np.testing.assert_allclose(samples.std(axis=0), SD, rtol=0.09)


def test_readme_gaussian_control_table():
    """README: endpoint/data std at 20 NFE leading is 0.86/0.73/0.44 (lambda 0) and 0.78/0.64/0.33 (lambda 1)."""
    grid = S.respaced_grid(T, 20)
    # Closed form: 0.8675 / 0.7254 / 0.4431 and 0.7829 / 0.6440 / 0.3293. The README's 0.86 is 0.87 when rounded.
    np.testing.assert_allclose(endpoint_law(grid, lam=0.0)[1] / SD, [0.86, 0.73, 0.44], atol=0.01)
    np.testing.assert_allclose(endpoint_law(grid, lam=1.0)[1] / SD, [0.78, 0.64, 0.33], atol=0.01)


@pytest.mark.parametrize('lam', [0.0, 0.5, 1.0])
def test_readme_finite_step_error_shrinks_with_nfe(lam):
    errors = [np.abs(endpoint_law(S.respaced_grid(T, nfe), lam=lam)[1] / SD - 1) for nfe in (20, 50, 100, 1000)]
    for coarse, fine in zip(errors, errors[1:]):
        assert (fine < coarse).all(), errors
