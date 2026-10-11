"""Regression tests for the v1 freeze: diagnostics, variance records, summary merging, paired
statistics, objective interface, preflight and the optional classifier metric."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from indep_helpers import (
    CLEAN, MARGIN, TINY_AB, TINY_B1, TINY_BT, TINY_STEPS, FakeEpsilon, np_beta_tilde, tiny_config, tiny_dataset,
)
from scaffold import loss as L
from scaffold import main as X
from scaffold import metrics as M
from scaffold import sampling as S
from scaffold.backbone import FixedScaffold, MLPScaffold

CPU = torch.device('cpu')
SCORE = 'fake-score-sha'
GRID = (40, 30, 20, 10, 0, CLEAN)


def metric(samples: Path, labels: np.ndarray):
    values = np.load(samples)
    return {'FID': {'Overall': float(values.mean()), 'Few': float(values.std())}}


def run_evaluate(config, output, **arguments):
    arguments.setdefault('epsilon', (FakeEpsilon(), SCORE))
    arguments.setdefault('metric_fn', metric)
    return X.evaluate(config, CPU, output=output, **arguments)


def edge_budgets():
    """lambda at which each stochastic edge of GRID hits the variance budget."""
    t, s = np.array(GRID[:-2]), np.array(GRID[1:-1])
    return (1 - MARGIN) * (1 - TINY_AB[s]) / np_beta_tilde(TINY_AB[t], TINY_AB[s])


# Requested vs effective variance

@pytest.mark.parametrize('lam', [0.0, 0.5, 1.0, 1.5, 4.0])
def test_transition_reports_requested_and_effective_variance(lam):
    schedule = S.Schedule.linear(TINY_STEPS, TINY_B1, TINY_BT)
    x = torch.randn(3, 3, 4, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    t, s = torch.full((3,), 40), torch.full((3,), 30)
    transition = S.scaffold_transition(x, torch.randn_like(x), lam, schedule.alpha_bar_at(t, x), schedule.alpha_bar_at(s, x))
    base = np_beta_tilde(TINY_AB[40], TINY_AB[30])
    budget = (1 - MARGIN) * (1 - TINY_AB[30])
    np.testing.assert_allclose(transition.requested.numpy(), lam * base, rtol=1e-7)
    np.testing.assert_allclose(transition.variance.numpy(), min(lam * base, budget), rtol=1e-7)
    assert bool(transition.capped.all()) == (lam * base > budget)


def test_evaluation_records_the_variance_each_arm_actually_used(tmp_path):
    config = tiny_config()
    config['evaluation']['arms'] = [{'name': 'ddim', 'kind': 'fixed', 'lambda': 0.0},
                                    {'name': 'wide', 'kind': 'fixed', 'lambda': 1.25},
                                    {'name': 'large', 'kind': 'ddpm', 'variance_type': 'fixedlarge'}]
    output = run_evaluate(config, tmp_path / 'panel')
    budgets = edge_budgets()
    assert budgets.min() < 1.25 < budgets.max(), 'the control needs both clipped and free edges'
    record = json.loads((output / 'seed_0' / 'wide' / 'variance.json').read_text())
    assert record['samples'] == 8 and [row['edge'] for row in record['edges']] == [list(e) for e in zip(GRID[:-2], GRID[1:-1])]
    for row, budget in zip(record['edges'], budgets):
        assert row['lambda_requested_mean'] == pytest.approx(1.25, rel=1e-5)
        assert row['lambda_effective_mean'] == pytest.approx(min(1.25, budget), rel=1e-5)
        assert row['variance_clip_fraction'] == float(1.25 > budget)
    ddim = json.loads((output / 'seed_0' / 'ddim' / 'variance.json').read_text())
    assert all(row['lambda_effective_mean'] == 0 and row['variance_clip_fraction'] == 0 for row in ddim['edges'])
    t, s = np.array(GRID[:-2]), np.array(GRID[1:-1])
    large = json.loads((output / 'seed_1' / 'large' / 'variance.json').read_text())
    expected = (1 - TINY_AB[t] / TINY_AB[s]) / np_beta_tilde(TINY_AB[t], TINY_AB[s])
    # float32 sampling: beta~ of the last edge is ~1e-5, so its ratio is only known to ~1e-4.
    np.testing.assert_allclose([row['lambda_effective_mean'] for row in large['edges']], expected, rtol=1e-3)
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['variance']['wide']['lambda_requested_mean'] == pytest.approx(1.25, rel=1e-5)
    assert summary['variance']['wide']['lambda_effective_mean'] == pytest.approx(np.minimum(1.25, budgets).mean(), rel=1e-5)
    assert 0 < summary['variance']['wide']['variance_clip_fraction'] < 1


def test_variance_records_survive_resume_identically(tmp_path):
    config = tiny_config()
    config['evaluation']['arms'] = [{'name': 'wide', 'kind': 'fixed', 'lambda': 1.25}]
    output = run_evaluate(config, tmp_path / 'panel')
    arm = output / 'seed_0' / 'wide'
    before = (arm / 'variance.json').read_text(), np.load(arm / 'samples.npy')
    (arm / 'samples.npy').unlink()
    (arm / 'variance.json').unlink()
    (arm / 'variance' / 'batch_000001.json').unlink()
    run_evaluate(config, output)
    assert (arm / 'variance.json').read_text() == before[0]
    np.testing.assert_array_equal(np.load(arm / 'samples.npy'), before[1])


# Summary merging and paired statistics

def test_partial_runs_extend_the_summary_instead_of_replacing_it(tmp_path):
    config = tiny_config()
    output = run_evaluate(config, tmp_path / 'panel', arms=['ddim', 'eta1'])
    first = json.loads((output / 'summary.json').read_text())
    run_evaluate(config, output, arms=['half'])
    merged = json.loads((output / 'summary.json').read_text())
    assert set(merged['per_seed']) == {'ddim', 'half', 'eta1'}, 'earlier arms are kept'
    for arm in ('ddim', 'eta1'):
        assert merged['per_seed'][arm] == first['per_seed'][arm] and merged['across_seeds'][arm] == first['across_seeds'][arm]
    one_seed = copy.deepcopy(config)
    one_seed['evaluation']['seeds'] = [1]
    run_evaluate(one_seed, output, arms=['large'])
    final = json.loads((output / 'summary.json').read_text())
    assert set(final['per_seed']) == {'ddim', 'half', 'eta1', 'large'}
    assert set(final['per_seed']['large']) == {'1'} and set(final['per_seed']['ddim']) == {'0', '1'}
    assert final['arms']['half'] == {'kind': 'fixed', 'lambda': 0.5}


def test_summary_ignores_reports_of_another_protocol(tmp_path):
    config = tiny_config()
    output = run_evaluate(config, tmp_path / 'panel', arms=['ddim', 'half'])
    report_path = output / 'seed_0' / 'half' / 'metrics.json'
    report = json.loads(report_path.read_text())
    report['binding']['protocol_sha256'] = 'another-protocol'
    report_path.write_text(json.dumps(report))
    protocol = json.loads((output / 'PROTOCOL.json').read_text())
    summary = X.collect_summary(output, protocol, config['evaluation'])
    assert set(summary['per_seed']['half']) == {1} and set(summary['per_seed']['ddim']) == {0, 1}


def test_paired_differences_are_per_seed_differences():
    rows = {'ddim': {0: {'FID': {'Overall': 10.0, 'Few': 30.0}}, 1: {'FID': {'Overall': 12.0, 'Few': 33.0}},
                     2: {'FID': {'Overall': 11.0, 'Few': 31.0}}},
            'arm': {0: {'FID': {'Overall': 9.0, 'Few': 31.0}}, 1: {'FID': {'Overall': 11.5, 'Few': 33.0}}},
            'extra': {5: {'FID': {'Overall': 1.0, 'Few': 1.0}}}}
    paired = X.paired_differences(rows, 'ddim')
    assert set(paired) == {'arm'}, 'the reference itself and arms without a common seed are left out'
    overall = paired['arm']['FID.Overall']
    assert overall['per_seed'] == {'0': -1.0, '1': -0.5} and overall['seeds'] == 2
    assert overall['mean'] == pytest.approx(-0.75) and overall['std'] == pytest.approx(np.std([-1.0, -0.5], ddof=1))
    assert paired['arm']['FID.Few']['mean'] == pytest.approx(0.5)


def test_summary_pairs_against_ddim_and_the_best_fixed_arm(tmp_path):
    output = run_evaluate(tiny_config(), tmp_path / 'panel')
    summary = json.loads((output / 'summary.json').read_text())
    fixed = {name: summary['across_seeds'][name]['FID.Overall']['mean'] for name in ('ddim', 'half', 'eta1')}
    best = min(fixed, key=fixed.get)
    assert summary['best_fixed_arm'] == {'name': best, 'selection_metric': 'FID.Overall', 'selected_on_reported_seeds': True}
    assert {'ddim', best} == set(summary['paired_differences'])
    for arm in ('half', 'eta1', 'small', 'large'):
        entry = summary['paired_differences']['ddim'][arm]['FID.Overall']
        deltas = [summary['per_seed'][arm][seed]['FID']['Overall'] - summary['per_seed']['ddim'][seed]['FID']['Overall']
                  for seed in ('0', '1')]
        assert entry['mean'] == pytest.approx(np.mean(deltas)) and entry['std'] == pytest.approx(np.std(deltas, ddof=1))
    assert 'ddim' not in summary['paired_differences']['ddim']


def test_learned_arm_records_its_training_seed(tmp_path):
    config = tiny_config()
    run = X.train(config, CPU, output=tmp_path / 'run', epsilon=(FakeEpsilon(), SCORE), data=tiny_dataset())
    config['evaluation']['arms'] = [{'name': 'phi', 'kind': 'learned', 'checkpoint': str(run / 'latest.pt')}]
    output = run_evaluate(config, tmp_path / 'panel')
    assert json.loads((output / 'seed_0' / 'phi' / 'ARM.json').read_text())['arm']['training_seed'] == 3
    payload = torch.load(run / 'latest.pt', weights_only=False)
    assert payload['version'] and set(payload['source_sha256']) >= {'sampling.py', 'loss.py', 'main.py', 'config.json'}


# Edge-level diagnostics

class TrueNoise:
    """A perfect denoiser for one batch: returns the noise that was added to x0 (plus an optional error)."""

    def __init__(self, x0, error=0.0):
        self.x0, self.error = x0, error

    def __call__(self, x, t, labels):
        alpha = torch.from_numpy(TINY_AB)[t].reshape(-1, 1, 1, 1)
        return (x - alpha.sqrt() * self.x0) / (1 - alpha).sqrt() + self.error


@pytest.fixture()
def bound():
    schedule = S.Schedule.linear(TINY_STEPS, TINY_B1, TINY_BT)
    x0 = torch.rand(12, 3, 8, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(1)) * 2 - 1
    return schedule, L.RespacedVLB(schedule, GRID), x0, torch.arange(12) % 4


def scaffold_at(initial):
    return MLPScaffold(4, image_size=8, hidden_dims=(16,), condition_dim=8, variance_initial=initial).double()


def test_edge_diagnostics_rows_match_the_bound_edge_by_edge(bound):
    schedule, loss, x0, labels = bound
    kernel = S.ScaffoldKernel(schedule, scaffold_at(0.5))
    epsilon = TrueNoise(x0, error=0.3)
    report = loss.edge_diagnostics(epsilon, kernel, x0, labels, generator=torch.Generator().manual_seed(4))
    rows = report['edges']
    assert [row['edge'] for row in rows] == [list(edge) for edge in zip(GRID, GRID[1:])]
    assert [row['is_terminal'] for row in rows] == [False] * 4 + [True]
    assert [row['edge_weight'] for row in rows] == [1.0] * 4 + [0.0]
    generator = torch.Generator().manual_seed(4)
    for index, row in enumerate(rows[:-1]):
        noise = torch.randn(x0.shape, dtype=x0.dtype, generator=generator)
        details = loss.edge_kl(epsilon, kernel, x0, labels, torch.full((12,), index), noise)
        assert row['edge_kl_nats_per_dim'] == pytest.approx(float(details['kl'].mean()) / 192, rel=1e-9)
        assert row['lambda_requested_mean'] == pytest.approx(0.5) and row['lambda_effective_mean'] == pytest.approx(0.5)
        assert row['variance_clip_fraction'] == 0 and row['edge_gradient_norm'] > 0
        assert row['lambda_budget'] == pytest.approx(edge_budgets()[index] / (1 - MARGIN), rel=1e-7)
    noise = torch.randn(x0.shape, dtype=x0.dtype, generator=generator)
    boundary = loss.boundary_terms(epsilon, kernel, x0, labels, noise)
    summary = report['summary']
    assert rows[-1]['edge_kl_nats_per_dim'] == pytest.approx(float(boundary['recon'].mean()) / 192)
    assert rows[-1]['edge_gradient_norm'] == 0 and rows[-1]['lambda_requested_mean'] is None
    trained = sum(row['edge_kl_nats_per_dim'] for row in rows[:-1])
    assert summary['trained_loss_nats_per_dim'] == pytest.approx(trained)
    assert summary['trained_loss_with_gradient_nats_per_dim'] == pytest.approx(trained)
    assert summary['full_bound_nats_per_dim'] == pytest.approx(
        trained + rows[-1]['edge_kl_nats_per_dim'] + float(boundary['prior'].mean()) / 192)
    dominant = max(rows, key=lambda row: row['edge_kl_nats_per_dim'])
    assert summary['dominant_edge'] == dominant['edge']
    assert summary['zero_gradient_edges'] == [] and summary['caveat'] == L.RespacedVLB.CAVEAT
    json.dumps(report)  # machine readable


def test_edge_diagnostics_expose_clipped_zero_gradient_edges(bound):
    schedule, loss, x0, labels = bound
    budgets = edge_budgets()
    assert budgets.max() < 20
    report = loss.edge_diagnostics(TrueNoise(x0, 0.3), S.ScaffoldKernel(schedule, scaffold_at(20.0)), x0, labels,
                                   generator=torch.Generator().manual_seed(4))
    for row, budget in zip(report['edges'][:-1], budgets):
        assert row['lambda_requested_mean'] == pytest.approx(20.0)
        assert row['lambda_effective_mean'] == pytest.approx(budget, rel=1e-7)
        assert row['variance_clip_fraction'] == 1.0 and row['edge_gradient_norm'] == 0.0
    summary = report['summary']
    assert summary['zero_gradient_edges'] == [list(edge) for edge in zip(GRID[:-2], GRID[1:-1])]
    assert summary['trained_loss_with_gradient_nats_per_dim'] == 0.0 < summary['trained_loss_nats_per_dim']


def test_fixed_lambda_sweep_locates_each_edges_optimum(bound):
    """With a perfect x0_hat lambda = 1 is exactly optimal. With a biased x0_hat this bound still never
    prefers lambda < 1 on any edge: the objective cannot reward a less stochastic sampler."""
    schedule, loss, x0, labels = bound
    kernel = S.ScaffoldKernel(schedule, FixedScaffold(1.0))
    perfect = loss.edge_diagnostics(TrueNoise(x0), kernel, x0, labels, generator=torch.Generator().manual_seed(4))
    assert all(row['best_fixed_lambda'] == 1.0 for row in perfect['edges'][:-1])
    assert all(row['fixed_lambda_kl_nats_per_dim']['1.0'] < 1e-9 for row in perfect['edges'][:-1])
    assert all(row['edge_gradient_norm'] == 0 for row in perfect['edges']), 'a fixed scaffold has no parameters'
    assert perfect['summary']['edges_preferring_lambda_below_one'] == []
    biased = loss.edge_diagnostics(TrueNoise(x0, 1.0), kernel, x0, labels, generator=torch.Generator().manual_seed(4))
    assert all(row['best_fixed_lambda'] >= 1.0 for row in biased['edges'][:-1])
    assert biased['summary']['edges_preferring_lambda_below_one'] == []
    for row in biased['edges'][:-1]:
        sweep = row['fixed_lambda_kl_nats_per_dim']
        assert set(sweep) == {str(value) for value in L.DIAGNOSTIC_LAMBDAS}
        assert sweep['1.0'] == pytest.approx(row['edge_kl_nats_per_dim'], rel=1e-9), 'same states, same kernel'
        assert sweep['0.05'] > sweep['0.25'] > sweep['0.5'] > sweep['1.0']


def test_diagnose_command_writes_a_report_for_initial_and_trained_scaffolds(tmp_path):
    config, epsilon, data = tiny_config(), (FakeEpsilon(), SCORE), tiny_dataset()
    path = X.diagnose(config, CPU, samples=16, output=tmp_path / 'diagnostics', epsilon=epsilon, data=data)
    report = json.loads(path.read_text())
    assert path.name == 'initial.json' and report['scaffold'] == 'initial' and report['objective'] == 'respaced_vlb'
    assert report['grid'] == list(GRID) and report['samples'] == 16 and len(report['edges']) == 5
    assert all(row['lambda_requested_mean'] == pytest.approx(1.0) for row in report['edges'][:-1])
    again = X.diagnose(config, CPU, samples=16, output=tmp_path / 'again', epsilon=epsilon, data=data)
    assert json.loads(again.read_text()) == report, 'deterministic'
    run = X.train(config, CPU, output=tmp_path / 'run', epsilon=epsilon, data=data)
    trained = X.diagnose(config, CPU, checkpoint=run / 'latest.pt', weights='online', samples=16,
                         output=tmp_path / 'diagnostics', epsilon=epsilon, data=data)
    assert trained.name == 'update_6_online.json'
    assert any(row['edge_gradient_norm'] > 0 for row in json.loads(trained.read_text())['edges'])


# Objective interface

def test_a_new_objective_plugs_in_without_touching_the_sampler(tmp_path, monkeypatch):
    calls = []

    class MeanMultiplier:
        """Toy objective: pull the effective multiplier of a random edge towards 0.4."""

        def __init__(self, schedule, grid):
            self.schedule, self.grid = schedule, grid

        def __call__(self, epsilon_fn, kernel, x0, labels, *, generator):
            t = torch.full_like(labels, self.grid[1])
            s = torch.full_like(labels, self.grid[2])
            transition = kernel(x0, epsilon_fn(x0, t, labels), t, s, labels)
            base = S.posterior_variance(self.schedule.alpha_bar_at(t, x0), self.schedule.alpha_bar_at(s, x0))
            multiplier = transition.variance / base
            calls.append(float(multiplier.mean()))
            return (multiplier - 0.4).square().mean(), {'multiplier': multiplier.detach(), 'capped': transition.capped}

    monkeypatch.setattr(L, 'OBJECTIVES', dict(L.OBJECTIVES))
    L.register_objective('mean_multiplier')(lambda settings, schedule, grid: MeanMultiplier(schedule, grid))
    with pytest.raises(ValueError, match='already registered'):
        L.register_objective('mean_multiplier')(lambda *_: None)
    with pytest.raises(ValueError, match='loss.kind'):
        L.build_loss({'kind': 'unknown'}, S.Schedule.linear(TINY_STEPS, TINY_B1, TINY_BT), GRID)
    config = tiny_config()
    config['loss'] = {'kind': 'mean_multiplier'}
    config['training'].update(updates=40, checkpoint_interval=40, learning_rate=3e-2, ema_decay=0.0)
    config['validation']['per_class'] = 1
    monkeypatch.setattr(X, 'validate', lambda *arguments, **keywords: {})
    X.train(config, CPU, output=tmp_path / 'run', epsilon=(FakeEpsilon(), SCORE), data=tiny_dataset())
    assert calls[0] == pytest.approx(1.0, abs=1e-4) and abs(calls[-1] - 0.4) < 0.15, 'lambda < 1 is learnable'
    assert isinstance(L.build_loss({'kind': 'respaced_vlb'}, S.Schedule.linear(TINY_STEPS, TINY_B1, TINY_BT), GRID),
                      L.RespacedVLB)


# Preflight

class GaussianDenoiser(torch.nn.Module):
    """Exact noise prediction for N(mean, std^2 I) data, with the vendor UNet call signature."""

    def __init__(self, mean, std, shift=0):
        super().__init__()
        self.register_buffer('alpha_bar', torch.from_numpy(TINY_AB).float())
        self.mean, self.std, self.shift = mean, std, shift

    def forward(self, x, t, y=None, return_mid=False):
        alpha = self.alpha_bar[(t + self.shift) % TINY_STEPS].reshape(-1, 1, 1, 1)
        out = (1 - alpha).sqrt() * (x - alpha.sqrt() * self.mean) / (alpha * self.std ** 2 + 1 - alpha)
        if y is not None:
            out = out + 0.01 * alpha.sqrt() * (y.float().reshape(-1, 1, 1, 1) - 1.5)
        return (out, out) if return_mid else out


def run_preflight(tmp_path, model):
    config, data = tiny_config(), tiny_dataset()
    model = model.requires_grad_(False).eval()
    path, passed = X.preflight(config, CPU, num_images=8, output=tmp_path / 'preflight', score=(model, SCORE),
                               data=data, metric_fn=metric)
    return json.loads(path.read_text()), passed


def data_moments():
    pixels = tiny_dataset()[0].astype(np.float64) / 127.5 - 1
    return float(pixels.mean()), float(pixels.std())


def test_preflight_passes_for_a_consistent_score_and_reports_every_check(tmp_path):
    report, passed = run_preflight(tmp_path, GaussianDenoiser(*data_moments()))
    checks = {entry['name']: entry for entry in report['checks']}
    assert list(checks) == ['score_checkpoint', 'noise_schedule', 'dataset_split', 'score_noise_prediction',
                            'ddim_parity', 'ddpm_parity', 'baseline_panel']
    assert passed and report['passed'] and all(entry['passed'] for entry in report['checks']), report
    assert set(checks['ddim_parity']['by_guidance']) == {'0.0', '1.5'}
    for row in checks['ddim_parity']['by_guidance'].values():
        assert row['max_step_error'] < 1e-5 and row['endpoint_error'] <= row['endpoint_allowed']
    for row in checks['score_noise_prediction']['timesteps'].values():
        assert row['beats_wrong_timesteps'] and row['quality_ok']
    assert [row['regime'] for row in checks['score_noise_prediction']['timesteps'].values()] == ['data'] * 3 + ['near_noise']
    assert max(checks['ddpm_parity']['max_abs_error_by_variance'].values()) < 1e-4
    assert checks['baseline_panel']['arms'] == ['ddim', 'eta1', 'small', 'large']
    assert set(report['source_sha256']) >= {'sampling.py', 'main.py'} and report['version']
    assert (tmp_path / 'preflight' / 'panel' / 'summary.json').is_file()


def test_preflight_fails_when_the_score_uses_another_timestep_convention(tmp_path):
    report, passed = run_preflight(tmp_path, GaussianDenoiser(*data_moments(), shift=20))
    checks = {entry['name']: entry for entry in report['checks']}
    assert not passed and not report['passed']
    assert not checks['score_noise_prediction']['passed']
    assert checks['ddim_parity']['passed'], 'parity with the vendor sampler cannot see a consistently wrong score'
    assert len(report['checks']) == 7, 'a failed check does not stop the remaining ones'


@pytest.mark.parametrize('shift', [5, -5])
def test_preflight_fails_on_the_convention_check_alone_for_a_timestep_offset(tmp_path, shift):
    """A score offset by 10 % of the schedule still passes the quality criteria; only the convention check sees it.

    Sensitivity limit, measured on this control: offsets of 2-3 indices in 50 (about 5 %) go unnoticed.
    Off-by-one conventions are the job of the parity checks, not of this one.
    """
    report, passed = run_preflight(tmp_path, GaussianDenoiser(*data_moments(), shift=shift))
    rows = {entry['name']: entry for entry in report['checks']}['score_noise_prediction']['timesteps']
    assert not passed
    assert all(row['quality_ok'] for row in rows.values()), 'the quality criteria alone would let this through'
    assert not all(row['beats_wrong_timesteps'] for row in rows.values())


def test_noise_prediction_verdict_on_the_measured_pretrained_values():
    """The criteria applied to the numbers measured with the pinned checkpoint on 2026-10-11 (CFG 0)."""
    measured = {50: (0.1025, {'200': 0.4554}, 0.9451), 300: (0.01157, {'150': 0.03, '450': 0.04174}, 0.2219),
                600: (0.002675, {'450': 0.004, '750': 0.004589}, 0.007604), 950: (4.047e-5, {'800': 0.4032}, 2.959e-5)}
    verdicts = {index: X.noise_prediction_verdict(index, 1000, *values) for index, values in measured.items()}
    assert [verdict['regime'] for verdict in verdicts.values()] == ['data', 'data', 'data', 'near_noise']
    assert all(verdict['beats_wrong_timesteps'] and verdict['quality_ok'] for verdict in verdicts.values())


def test_noise_prediction_verdict_separates_convention_from_quality():
    verdict = X.noise_prediction_verdict
    good = verdict(300, 1000, 0.01, {'150': 0.03, '450': 0.04}, 0.2)
    assert good['beats_wrong_timesteps'] and good['quality_ok']
    # Convention: a single better wrong timestep is enough to fail, whatever the quality.
    assert not verdict(300, 1000, 0.01, {'150': 0.009, '450': 0.04}, 0.2)['beats_wrong_timesteps']
    assert verdict(300, 1000, 0.01, {'150': 0.009, '450': 0.04}, 0.2)['quality_ok']
    # Quality in the data regime: must beat the trivial predictor outright, up to and including 0.6 T.
    assert not verdict(600, 1000, 0.01, {'450': 0.03}, 0.009)['quality_ok']
    assert verdict(600, 1000, 0.01, {'450': 0.03}, 0.009)['beats_wrong_timesteps']
    # Near pure noise: within a factor 10 of the analytic predictor, not better than it.
    assert verdict(950, 1000, 4e-5, {'800': 0.4}, 3e-5)['quality_ok']
    assert not verdict(950, 1000, 4e-4, {'800': 0.4}, 3e-5)['quality_ok']
    assert verdict(601, 1000, 4e-5, {'451': 0.4}, 3e-5)['regime'] == 'near_noise'


@pytest.mark.parametrize('failing', ['beats_wrong_timesteps', 'quality_ok'])
def test_preflight_score_check_needs_both_verdicts_at_every_noise_level(tmp_path, monkeypatch, failing):
    original, calls = X.noise_prediction_verdict, []

    def one_bad_level(index, *arguments):
        verdict = original(index, *arguments)
        calls.append(index)
        return {**verdict, failing: False} if len(calls) == 3 else verdict

    monkeypatch.setattr(X, 'noise_prediction_verdict', one_bad_level)
    report, passed = run_preflight(tmp_path, GaussianDenoiser(*data_moments()))
    checks = {entry['name']: entry for entry in report['checks']}
    assert len(calls) == 4 and not passed and not checks['score_noise_prediction']['passed']
    assert checks['ddim_parity']['passed'] and checks['baseline_panel']['passed']


def test_preflight_endpoint_check_catches_a_sampling_loop_error_the_step_check_cannot_see(tmp_path, monkeypatch):
    """Each edge is right, but the loop silently drops one node of the grid: only the endpoint differs."""
    original = X.sample

    def skipping(epsilon_fn, kernel, labels, initial_noise, grid, **keywords):
        shortened = grid[:2] + grid[3:] if len(grid) > 3 else grid
        return original(epsilon_fn, kernel, labels, initial_noise, shortened, **keywords)

    monkeypatch.setattr(X, 'sample', skipping)
    report, passed = run_preflight(tmp_path, GaussianDenoiser(*data_moments()))
    rows = {entry['name']: entry for entry in report['checks']}['ddim_parity']['by_guidance']
    assert not passed
    assert all(row['step_ok'] for row in rows.values()) and not any(row['endpoint_ok'] for row in rows.values())


@pytest.mark.parametrize('factor', [1.001, 0.999])
def test_preflight_parity_fails_for_a_slightly_wrong_update_rule(tmp_path, monkeypatch, factor):
    """A 0.1 % error in the mean of every edge must fail both parity checks, per step and at the endpoint."""
    original = S.scaffold_transition

    def skewed(*arguments, **keywords):
        transition = original(*arguments, **keywords)
        return S.Transition(transition.x0_hat, factor * transition.mean, transition.variance,
                            transition.capped, transition.requested)

    monkeypatch.setattr(S, 'scaffold_transition', skewed)
    report, passed = run_preflight(tmp_path, GaussianDenoiser(*data_moments()))
    checks = {entry['name']: entry for entry in report['checks']}
    assert not passed and not checks['ddim_parity']['passed'] and not checks['ddpm_parity']['passed']
    assert all(not row['step_ok'] for row in checks['ddim_parity']['by_guidance'].values())
    assert checks['score_noise_prediction']['passed'], 'the score itself is untouched'


def test_preflight_reports_a_missing_checkpoint_and_its_dependents(tmp_path):
    config = tiny_config()
    config['paths']['checkpoint'] = str(tmp_path / 'absent.pt')
    path, passed = X.preflight(config, CPU, num_images=8, output=tmp_path / 'preflight', data=tiny_dataset(), metric_fn=metric)
    checks = {entry['name']: entry for entry in json.loads(path.read_text())['checks']}
    assert not passed and 'FileNotFoundError' in checks['score_checkpoint']['error']
    assert checks['noise_schedule']['passed'] and checks['dataset_split']['passed']
    assert "prerequisite check 'score_checkpoint' failed" in checks['ddim_parity']['error']


# Optional classifier accuracy

class BrightnessClassifier(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bucket = (x.mean(dim=(1, 2, 3)) * 100).clamp(0, 99).long()
        return torch.nn.functional.one_hot(bucket, 100).float()


def test_classifier_accuracy_overall_and_per_group(tmp_path):
    path = tmp_path / 'classifier.pt'
    torch.jit.script(BrightnessClassifier()).save(str(path))
    settings = {'torchscript': str(path), 'sha256': M.sha256_file(path), 'mean': [0.0, 0.0, 0.0],
                'std': [1.0, 1.0, 1.0], 'batch_size': 64}
    labels = np.arange(200) % 100
    predicted = labels.copy()
    predicted[labels >= 66] = 0  # every Few-class sample is recognised as class 0
    images = np.broadcast_to(((predicted + 0.5) / 100).astype(np.float32)[:, None, None, None], (200, 8, 8, 3))
    accuracy = M.classifier_accuracy(images, labels, settings, CPU)
    assert accuracy == {'Overall': pytest.approx(0.66), 'Many': 1.0, 'Medium': 1.0, 'Few': 0.0}
    with pytest.raises(RuntimeError, match='SHA mismatch'):
        M.classifier_accuracy(images, labels, {**settings, 'sha256': 'wrong'}, CPU)


# Quadratic timestep grid (DDIM reference code)

@pytest.mark.parametrize('nfe', [10, 20, 50, 100])
def test_quadratic_grid_follows_the_ddim_reference_selection(nfe):
    reference = [int(value) for value in list(np.linspace(0, np.sqrt(1000 * 0.8), nfe) ** 2)]  # DDIM runners/diffusion.py
    grid = S.respaced_grid(1000, nfe, spacing='quadratic')
    assert grid[-1] == CLEAN and len(grid) == nfe + 1, 'exactly K evaluations of theta'
    nodes = list(grid[:-1])[::-1]
    assert all(b > a for a, b in zip(nodes, nodes[1:])) and nodes[0] == 0 and nodes[-1] in (799, 800)
    collisions = len(reference) - len(set(reference))
    if collisions == 0:
        assert nodes == reference
    else:
        # Colliding indices are pushed up by one each; the shift never exceeds the number of collisions.
        assert all(0 <= a - b <= collisions for a, b in zip(nodes, reference))
    assert S.build_grid(1000, {'nfe': nfe, 'spacing': 'quadratic'}) == grid
