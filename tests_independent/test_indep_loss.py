"""loss.py against Bayes' rule, torch.distributions and scipy."""
import numpy as np
import pytest
import torch
from scipy import stats

from indep_helpers import CLEAN, ab_at, np_alpha_bar, np_edge, np_gaussian_kl, np_posterior_bayes
from scaffold import loss as L
from scaffold import sampling as S
from scaffold.backbone import FixedScaffold, MLPScaffold

T = 1000
AB = np_alpha_bar()
SHAPE = (3, 8, 8)
DIMENSIONS = int(np.prod(SHAPE))


@pytest.fixture(scope='module')
def schedule():
    return S.Schedule.linear(T, 1e-4, 0.02)


@pytest.fixture(scope='module')
def bound(schedule):
    return L.RespacedVLB(schedule, S.respaced_grid(T, 20))


def column(values):
    return torch.as_tensor(np.asarray(values), dtype=torch.float64).reshape(-1, 1, 1, 1)


def pixels(rng, batch):
    """Images on the 8-bit grid scaled to [-1, 1], including both edge values."""
    values = rng.integers(0, 256, size=(batch, *SHAPE))
    values[0, 0, 0, :4] = [0, 255, 0, 255]
    return values / 127.5 - 1.0


# Gaussian helpers

def test_gaussian_kl_matches_torch_distributions():
    generator = torch.Generator().manual_seed(0)
    mean_q, mean_p = torch.randn(5, 3, 4, 4, generator=generator), torch.randn(5, 3, 4, 4, generator=generator)
    var_q = torch.rand(5, 3, 4, 4, generator=generator) + 0.05
    var_p = torch.rand(5, 3, 4, 4, generator=generator) + 0.05
    expected = torch.distributions.kl_divergence(torch.distributions.Normal(mean_q, var_q.sqrt()),
                                                 torch.distributions.Normal(mean_p, var_p.sqrt())).flatten(1).sum(1)
    torch.testing.assert_close(L.gaussian_kl(mean_q, var_q, mean_p, var_p), expected, rtol=1e-5, atol=1e-5)
    assert L.gaussian_kl(mean_q, var_q, mean_q, var_q).abs().max() == 0
    assert (L.gaussian_kl(mean_q, var_q, mean_p, var_p) > 0).all()


@pytest.mark.parametrize('var_q,var_p', [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
def test_gaussian_kl_rejects_nonpositive_variance(var_q, var_p):
    one = torch.ones(2, 3)
    with pytest.raises(ValueError):
        L.gaussian_kl(one, one * var_q, one, one * var_p)


@pytest.mark.parametrize('mean,variance', [(0.1, 0.01 ** 2), (-0.4, 0.05 ** 2), (0.97, 0.2 ** 2), (0.0, 1.0)])
def test_discretized_decoder_is_a_distribution_over_256_values(mean, variance):
    values = torch.arange(256, dtype=torch.float64).reshape(256, 1) / 127.5 - 1.0
    log_probability = L.discretized_gaussian_log_likelihood(values, torch.full_like(values, mean),
                                                            torch.full_like(values, variance))
    assert log_probability.shape == (256,)
    assert log_probability.exp().sum().item() == pytest.approx(1.0, abs=1e-6)


def test_discretized_decoder_matches_scipy():
    rng = np.random.default_rng(1)
    x0 = pixels(rng, 4)
    mean = np.clip(x0 + rng.normal(scale=0.05, size=x0.shape), -1.2, 1.2)
    variance = rng.uniform(0.01, 0.2, size=x0.shape) ** 2
    scale = np.sqrt(variance)
    upper, lower = stats.norm.cdf((x0 - mean + 1 / 255) / scale), stats.norm.cdf((x0 - mean - 1 / 255) / scale)
    probability = np.where(x0 < -0.999, upper, np.where(x0 > 0.999, 1 - lower, upper - lower))
    expected = np.log(np.maximum(probability, 1e-12)).reshape(4, -1).sum(1)
    got = L.discretized_gaussian_log_likelihood(torch.from_numpy(x0), torch.from_numpy(mean), torch.from_numpy(variance))
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-9)


def test_forward_sample_and_posterior_match_bayes_rule():
    rng = np.random.default_rng(2)
    t = rng.integers(1, T, size=6)
    s = np.array([rng.integers(0, value) for value in t])
    ab_t, ab_s = ab_at(AB, t), ab_at(AB, s)
    x0, noise = rng.normal(size=(6, *SHAPE)), rng.normal(size=(6, *SHAPE))
    x_t = np.sqrt(ab_t) * x0 + np.sqrt(1 - ab_t) * noise
    np.testing.assert_allclose(L.forward_sample(torch.from_numpy(x0), column(ab_t), torch.from_numpy(noise)).numpy(),
                               x_t, rtol=1e-12)
    mean, variance = np_posterior_bayes(x0, x_t, ab_t, ab_s)
    got_mean, got_variance = L.forward_posterior(torch.from_numpy(x0), torch.from_numpy(x_t), column(ab_t), column(ab_s))
    np.testing.assert_allclose(got_mean.numpy(), mean, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(got_variance.numpy(), variance, rtol=1e-8)


def test_forward_posterior_is_empirically_the_conditional_of_the_markov_chain():
    """Simulate x_s then x_t from x0 and regress: independent of every closed form above."""
    rng = np.random.default_rng(3)
    ab_t, ab_s, x0, count = AB[600], AB[400], 0.7, 400_000
    x_s = np.sqrt(ab_s) * x0 + np.sqrt(1 - ab_s) * rng.normal(size=count)
    x_t = np.sqrt(ab_t / ab_s) * x_s + np.sqrt(1 - ab_t / ab_s) * rng.normal(size=count)
    slope, intercept = np.polyfit(x_t, x_s, 1)
    residual_variance = np.var(x_s - (slope * x_t + intercept))
    probe = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    mean, variance = L.forward_posterior(torch.full_like(probe, x0), probe, torch.tensor(ab_t), torch.tensor(ab_s))
    assert mean[0].item() == pytest.approx(intercept, abs=3e-3)
    assert (mean[1] - mean[0]).item() == pytest.approx(slope, abs=3e-3)
    assert variance.item() == pytest.approx(residual_variance, rel=1e-2)


# The bound

def test_edges_are_the_noisy_edges_of_the_grid(schedule, bound):
    grid = S.respaced_grid(T, 20)
    assert bound.num_edges == 19
    assert bound.edge_t.tolist() == list(grid[:-2]) and bound.edge_s.tolist() == list(grid[1:-1])
    assert CLEAN not in bound.edge_s.tolist()


def test_bound_rejects_grids_without_a_noisy_edge_or_clean_end(schedule):
    with pytest.raises(ValueError):
        L.RespacedVLB(schedule, S.stride_grid(T, 10))
    with pytest.raises(ValueError):
        L.RespacedVLB(schedule, S.respaced_grid(T, 1))
    assert L.RespacedVLB(schedule, S.respaced_grid(T, 2)).num_edges == 1
    with pytest.raises(ValueError):
        L.build_loss({'kind': 'elbo'}, schedule, S.respaced_grid(T, 20))
    assert isinstance(L.build_loss({'kind': 'respaced_vlb'}, schedule, S.respaced_grid(T, 20)), L.RespacedVLB)


def test_stratified_edges_are_balanced_within_a_batch_and_uniform_marginally(bound):
    generator = torch.Generator().manual_seed(0)
    assert torch.bincount(bound.stratified_edges(38, generator), minlength=19).tolist() == [2] * 19
    partial = bound.stratified_edges(10, generator)
    assert len(set(partial.tolist())) == 10 and 0 <= int(partial.min()) and int(partial.max()) < 19
    counts = np.zeros(19)
    for _ in range(1900):
        counts += np.bincount(bound.stratified_edges(5, generator).numpy(), minlength=19)
    assert stats.chisquare(counts).pvalue > 1e-4, counts
    first = [int(bound.stratified_edges(19, generator)[0]) for _ in range(400)]
    assert len(set(first)) == 19, 'the within-batch position must not determine the edge'


class TrueNoise:
    """epsilon_fn that returns the exact forward noise, plus an optional error."""

    def __init__(self, x0, error=0.0):
        self.x0, self.error, self.alpha = x0, error, torch.from_numpy(AB)

    def __call__(self, x_t, t, labels):
        alpha = self.alpha[t].reshape(-1, 1, 1, 1)
        return (x_t - alpha.sqrt() * self.x0) / (1 - alpha).sqrt() + self.error


@pytest.mark.parametrize('lam', [0.2, 1.0, 1.6, 2.0, 40.0])
@pytest.mark.parametrize('error', [0.0, 0.3])
def test_edge_kl_matches_an_independent_bayes_computation(schedule, bound, lam, error):
    rng = np.random.default_rng(int(lam * 10))
    batch = 19
    x0, noise = pixels(rng, batch), rng.normal(size=(batch, *SHAPE))
    edges = np.arange(batch)
    t, s = np.array(bound.grid)[edges], np.array(bound.grid)[edges + 1]
    ab_t, ab_s = ab_at(AB, t), ab_at(AB, s)
    x_t = np.sqrt(ab_t) * x0 + np.sqrt(1 - ab_t) * noise
    _, mean_p, variance_p, capped = np_edge(x_t, noise + error, lam, ab_t, ab_s)
    mean_q, variance_q = np_posterior_bayes(x0, x_t, ab_t, ab_s)
    expected = np_gaussian_kl(mean_q, variance_q, mean_p, variance_p).reshape(batch, -1).sum(1)

    x0_tensor = torch.from_numpy(x0)
    kernel = S.ScaffoldKernel(schedule, FixedScaffold(lam))
    details = bound.edge_kl(TrueNoise(x0_tensor, error), kernel, x0_tensor, torch.zeros(batch, dtype=torch.long),
                            torch.from_numpy(edges), torch.from_numpy(noise))
    # 1 - ab_0 = 1e-4 amplifies the 1e-7 float32-beta difference between oracle and repo schedules.
    np.testing.assert_allclose(details['kl'].numpy(), expected, rtol=1e-4, atol=1e-6)
    assert details['t'].tolist() == t.tolist()
    np.testing.assert_allclose(details['multiplier'].numpy(), np.broadcast_to(variance_p / variance_q, x0.shape), rtol=1e-4)
    np.testing.assert_array_equal(details['capped'].numpy(), np.broadcast_to(capped, x0.shape))
    if lam == 1.0 and error == 0.0:
        assert details['kl'].abs().max() < 1e-6, 'a perfect score with the posterior variance has zero KL'
    else:
        assert (details['kl'] > 0).all()


def test_loss_is_an_unbiased_estimate_of_the_sum_over_edges(schedule, bound):
    """x0 = 0 makes x_t / sqrt(1 - ab_t) the exact noise, so E[KL_k] is available in closed form."""
    lam, batch = 0.4, 19 * 160
    t, s = np.array(bound.grid[:-2]), np.array(bound.grid[1:-1])
    beta_tilde = (1 - AB[s]) / (1 - AB[t]) * (1 - AB[t] / AB[s])
    shift = np.sqrt(1 - AB[s] - lam * beta_tilde) - np.sqrt(1 - AB[s] - beta_tilde)
    expected = np.sum(0.5 * (1 / lam - 1 + np.log(lam)) + 0.5 * shift ** 2 / (lam * beta_tilde))

    alpha = torch.from_numpy(AB)
    epsilon = lambda x, step, labels: x / (1 - alpha[step]).sqrt().reshape(-1, 1, 1, 1)
    x0 = torch.zeros(batch, *SHAPE, dtype=torch.float64)
    value, details = bound(epsilon, S.ScaffoldKernel(schedule, FixedScaffold(lam)), x0,
                           torch.zeros(batch, dtype=torch.long), generator=torch.Generator().manual_seed(4))
    assert value.item() == pytest.approx(expected, rel=0.01)
    assert details['per_sample'].shape == (batch,)
    assert value.item() == pytest.approx(details['per_sample'].mean().item(), rel=1e-12)
    np.testing.assert_allclose(details['per_sample'].numpy(), 19 * details['kl'].detach().numpy() / DIMENSIONS, rtol=1e-12)
    assert torch.bincount(torch.bucketize(details['t'], torch.tensor(sorted(t)))).tolist() == [160] * 19


def test_loss_is_reproducible_from_its_generator(schedule, bound):
    x0 = torch.from_numpy(pixels(np.random.default_rng(5), 8)).float()
    kernel = S.ScaffoldKernel(schedule, FixedScaffold(0.5))
    run = lambda seed: bound(TrueNoise(x0, 0.1), kernel, x0, torch.zeros(8, dtype=torch.long),
                             generator=torch.Generator().manual_seed(seed))[0].item()
    assert run(1) == run(1) and run(1) != run(2)


def test_gradient_reaches_phi_and_points_towards_lower_kl(schedule, bound):
    """With a perfect score the optimum is lambda = 1, so starting at 0.3 the loss must fall along -grad."""
    rng = np.random.default_rng(6)
    x0 = torch.from_numpy(pixels(rng, 38)).float()
    scaffold = MLPScaffold(10, image_size=8, hidden_dims=(16,), condition_dim=8, variance_initial=0.3)
    kernel = S.ScaffoldKernel(schedule, scaffold)
    labels = torch.arange(38) % 10
    evaluate = lambda: bound(TrueNoise(x0), kernel, x0, labels, generator=torch.Generator().manual_seed(7))
    value, details = evaluate()
    assert not details['capped'].any()
    value.backward()
    gradients = [parameter.grad for parameter in scaffold.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert float(scaffold.output.bias.grad.abs().max()) > 0
    assert float(scaffold.output.bias.grad.max()) <= 0, 'every pixel should be pushed towards a larger lambda'
    with torch.no_grad():
        scaffold.output.bias -= 0.2 * scaffold.output.bias.grad.sign()
    assert evaluate()[0].item() < value.item()


def cap_threshold(grid):
    """lambda above which an edge's variance is clipped by the budget (1 - 1e-4)(1 - ab_s)."""
    t, s = np.array(grid[:-2]), np.array(grid[1:-1])
    return (1 - 1e-4) * (1 - AB[t]) * AB[s] / (AB[s] - AB[t])


def test_budget_thresholds_of_the_default_grid_are_what_the_report_states():
    """Characterisation. On the config grid lambda is only free up to 1.003 (edge 50->0) ... 2.55 (edge 350->300).

    Consequences for the panel: the 'fixed_1.5' arm is clipped on 2 of 19 edges and 'fixed_2.0' on 8 of 19,
    and lambda_initial = 1 starts 0.3 % below the clip of the last noisy edge.
    """
    threshold = cap_threshold(S.respaced_grid(T, 20))
    assert threshold.min() == pytest.approx(1.003, abs=2e-3) and threshold.argmin() == 18
    assert threshold.max() == pytest.approx(2.548, abs=5e-3)
    assert int((threshold < 1.5).sum()) == 2 and int((threshold < 2.0).sum()) == 8


def test_capped_coordinates_receive_exactly_zero_gradient(schedule, bound):
    """Characterisation of a training hazard: the budget is a hard min, so phi gets no signal once clipped."""
    rng = np.random.default_rng(9)
    x0 = torch.from_numpy(pixels(rng, 38)).float()
    scaffold = MLPScaffold(10, image_size=8, hidden_dims=(16,), condition_dim=8, variance_initial=3.0)
    value, details = bound(TrueNoise(x0), S.ScaffoldKernel(schedule, scaffold), x0, torch.arange(38) % 10,
                           generator=torch.Generator().manual_seed(7))
    assert details['capped'].all(), 'lambda = 3 exceeds the budget on every edge of the 20-NFE grid'
    value.backward()
    assert all(float(parameter.grad.abs().max()) == 0.0 for parameter in scaffold.parameters()
               if parameter.grad is not None)


def test_boundary_terms_match_scipy_and_do_not_depend_on_phi(schedule, bound):
    rng = np.random.default_rng(8)
    batch = 5
    x0, noise = pixels(rng, batch), rng.normal(size=(batch, *SHAPE))
    first, top = bound.grid[-2], bound.grid[0]
    assert (first, top) == (0, 950)
    ab_first, ab_top, error = AB[first], AB[top], 0.5
    x_first = np.sqrt(ab_first) * x0 + np.sqrt(1 - ab_first) * noise
    x0_hat = (x_first - np.sqrt(1 - ab_first) * (noise + error)) / np.sqrt(ab_first)
    scale = np.sqrt(1 - ab_first)
    upper, lower = stats.norm.cdf((x0 - x0_hat + 1 / 255) / scale), stats.norm.cdf((x0 - x0_hat - 1 / 255) / scale)
    probability = np.where(x0 < -0.999, upper, np.where(x0 > 0.999, 1 - lower, upper - lower))
    recon = -np.log(np.maximum(probability, 1e-12)).reshape(batch, -1).sum(1)
    prior = np_gaussian_kl(np.sqrt(ab_top) * x0, 1 - ab_top, 0.0, 1.0).reshape(batch, -1).sum(1)

    x0_tensor, labels = torch.from_numpy(x0), torch.zeros(batch, dtype=torch.long)
    results = [bound.boundary_terms(TrueNoise(x0_tensor, error), S.ScaffoldKernel(schedule, FixedScaffold(lam)),
                                    x0_tensor, labels, torch.from_numpy(noise)) for lam in (0.0, 1.0, 5.0)]
    for result in results:
        np.testing.assert_allclose(result['recon'].numpy(), recon, rtol=1e-6)
        np.testing.assert_allclose(result['prior'].numpy(), prior, rtol=1e-6)
    assert (results[0]['recon'] > 0).all() and (results[0]['prior'] > 0).all()
    assert float(results[0]['prior'].max()) / DIMENSIONS < 1e-3, 'q(x_950 | x0) is almost the N(0, I) prior'


@pytest.mark.parametrize('spacing,dominant', [('leading', True), ('trailing', False)])
def test_share_of_the_last_noisy_edge_in_the_bound(spacing, dominant):
    """Characterisation, exact Gaussian score, data std 0.2, lambda = 1 (closed form, no sampling).

    KL_k = A_k^2 Var(x0 | x_t) / (2 beta~_k) with A_k = sqrt(ab_s) (1 - ab_t / ab_s) / (1 - ab_t).
    With 'leading' spacing the last noisy edge 50 -> 0 targets a state with noise std 0.01, so its
    beta~ is 1e-4 and it carries almost the whole bound, while its lambda is clipped at 1.003:
    most of the reported loss is a term phi cannot change.
    """
    grid = S.respaced_grid(T, 20, spacing=spacing)
    t, s = np.array(grid[:-2]), np.array(grid[1:-1])
    ratio, data_variance = AB[t] / AB[s], 0.2 ** 2
    slope = np.sqrt(AB[s]) * (1 - ratio) / (1 - AB[t])
    posterior_x0 = data_variance * (1 - AB[t]) / (AB[t] * data_variance + 1 - AB[t])
    kl = slope ** 2 * posterior_x0 / (2 * (1 - AB[s]) / (1 - AB[t]) * (1 - ratio))
    share = kl[-1] / kl.sum()
    print(f'{spacing}: last noisy edge {t[-1]}->{s[-1]} KL = {kl[-1]:.2f} nats/dim, {100 * share:.1f} % of the bound')
    assert (share > 0.9) == dominant
