"""main.py end to end on a tiny CPU problem, checked against an independent numpy sampler."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from indep_helpers import (CLEAN, TINY_AB, FakeEpsilon, ab_at, np_edge, np_fake_epsilon, protocol_generator,
                           tiny_config, tiny_dataset)
from scaffold import main as X
from scaffold.backbone import FixedScaffold, MLPScaffold
from scaffold.sampling import ScaffoldKernel, Schedule, build_grid

CPU = torch.device('cpu')
SCORE = 'sha-of-the-fake-score'
GRID = (40, 30, 20, 10, 0, CLEAN)


def run_train(config, output, **arguments):
    arguments.setdefault('epsilon', (FakeEpsilon(), SCORE))
    arguments.setdefault('data', tiny_dataset())
    return X.train(config, CPU, output=output, **arguments)


def mean_metric(calls=None):
    def metric(samples: Path, labels: np.ndarray):
        if calls is not None:
            calls.append(samples)
        values = np.load(samples)
        return {'FID': {'Overall': float(values.mean()), 'Few': float(values.std())}, 'count': len(labels)}
    return metric


def run_evaluate(config, output, **arguments):
    arguments.setdefault('epsilon', (FakeEpsilon(), SCORE))
    arguments.setdefault('metric_fn', mean_metric())
    return X.evaluate(config, CPU, output=output, **arguments)


# Small pieces

def test_draw_batch_is_class_uniform_with_correct_labels_and_flips():
    images, indices, _ = tiny_dataset()
    settings = {'samples_per_class': 4, 'classes_per_batch': 2, 'horizontal_flip': True}
    rng = np.random.default_rng(0)
    class_counts, flipped, total = np.zeros(4), 0, 0
    for _ in range(400):
        clean, labels = X.draw_batch(images, indices, settings, rng, CPU)
        assert clean.shape == (8, 3, 8, 8) and labels.dtype == torch.long
        assert labels[:4].unique().numel() == 1 and labels[4:].unique().numel() == 1 and labels[0] != labels[4]
        assert float(clean.min()) >= -1 and float(clean.max()) <= 1
        class_counts += np.bincount(labels.numpy(), minlength=4)
        for image, label in zip(clean, labels):
            candidates = torch.from_numpy(images[indices[int(label)]]).float().permute(0, 3, 1, 2) / 127.5 - 1
            straight = (candidates - image).abs().flatten(1).max(1).values.min() < 1e-6
            mirrored = (candidates.flip(-1) - image).abs().flatten(1).max(1).values.min() < 1e-6
            assert straight or mirrored, 'every drawn image belongs to its labelled class'
            flipped += int(mirrored and not straight)
            total += 1
    assert np.abs(class_counts / class_counts.sum() - 0.25).max() < 0.03, 'classes are uniform despite 12/8/5/2 images'
    assert 0.45 < flipped / total < 0.55
    plain, _ = X.draw_batch(images, indices, {**settings, 'horizontal_flip': False}, np.random.default_rng(1), CPU)
    originals = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 127.5 - 1
    assert all((originals - image).abs().flatten(1).max(1).values.min() < 1e-6 for image in plain)


def test_validation_panels_are_fixed_and_class_complete():
    images, indices, _ = tiny_dataset()
    config = tiny_config()
    heldout = (images[::-1].copy(), np.repeat(np.arange(4), [12, 8, 5, 2])[::-1].copy())
    panels = X.validation_panels(config, images, indices, heldout)
    again = X.validation_panels(config, images, indices, heldout)
    assert set(panels) == {'train', 'heldout'}
    for name in panels:
        np.testing.assert_array_equal(panels[name][0], again[name][0])
        assert np.bincount(panels[name][1], minlength=4).tolist() == [3, 3, 3, 2]
    for image, label in zip(*panels['train']):
        assert any((images[i] == image).all() for i in indices[label])
    assert set(X.validation_panels(config, images, indices, None)) == {'train'}


def test_ema_update_formula():
    online, average = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    before = [parameter.clone() for parameter in average.parameters()]
    X.update_ema(average, online, 0.9)
    for old, new, source in zip(before, average.parameters(), online.parameters()):
        torch.testing.assert_close(new, 0.9 * old + 0.1 * source)


def test_to_unit_nhwc_clamps_and_reorders():
    images = torch.zeros(1, 3, 2, 2)
    images[0, :, 0, 1] = torch.tensor([-3.0, 0.0, 5.0])
    values = X.to_unit_nhwc(images)
    assert values.shape == (1, 2, 2, 3) and values.dtype == np.float32 and values.flags['C_CONTIGUOUS']
    assert values[0, 0, 1].tolist() == [0.0, 0.5, 1.0] and values[0, 1, 1].tolist() == [0.5, 0.5, 0.5]


def test_summarize_reports_mean_and_sample_std_per_flattened_metric():
    rows = {'a': {0: {'FID': {'Overall': 1.0, 'Few': 4.0}, 'Recall': 0.5},
                  1: {'FID': {'Overall': 3.0, 'Few': 8.0}, 'Recall': 0.7}},
            'b': {0: {'FID': {'Overall': 2.0}}}}
    summary = X.summarize(rows)
    assert summary['a']['FID.Overall'] == {'mean': 2.0, 'std': pytest.approx(np.sqrt(2.0)), 'seeds': 2}
    assert summary['a']['FID.Few']['mean'] == 6.0 and summary['a']['Recall']['mean'] == pytest.approx(0.6)
    assert summary['b']['FID.Overall'] == {'mean': 2.0, 'std': None, 'seeds': 1}


def test_training_identity_excludes_only_evaluation():
    config = tiny_config()
    identity = X.training_identity(config)
    assert set(identity) == set(config) - {'evaluation'}


def test_panel_bound_is_deterministic_and_zero_for_a_perfect_posterior_sampler():
    config = tiny_config()
    schedule = Schedule.from_config(config)
    loss = X.build_loss(config['loss'], schedule, build_grid(len(schedule), config['sampler']['grid']))
    images, indices, _ = tiny_dataset()
    panel = X.validation_panels(config, images, indices, None)['train']
    kernel = ScaffoldKernel(schedule, FixedScaffold(1.0))
    run = lambda fn: X.panel_bound(loss, fn, kernel, panel, seed=5, batch_size=4, device=CPU)
    first = run(FakeEpsilon())
    assert first == run(FakeEpsilon()) and first['Overall'] > 0 and set(first) == {'Overall', 'Many'}
    other = X.panel_bound(loss, FakeEpsilon(), kernel, panel, seed=5, batch_size=11, device=CPU)
    assert other['Overall'] == pytest.approx(first['Overall'], rel=1e-5), 'the bound must not depend on the batch size'


# Training

@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    output = tmp_path_factory.mktemp('train') / 'run'
    run_train(tiny_config(), output)
    return output


def test_train_writes_the_documented_outputs(trained):
    names = sorted(path.name for path in trained.iterdir())
    assert names == ['COMPLETE.train.json', 'config.json', 'latest.pt', 'reference_lambda1.json', 'snapshot_3.pt',
                     'snapshot_6.pt', 'train.jsonl']
    rows = [json.loads(line) for line in (trained / 'train.jsonl').read_text().splitlines()]
    assert [row['update'] for row in rows] == [1, 2, 3, 4, 5, 6]
    assert ['validation' in row for row in rows] == [False, False, True, False, False, True]
    assert all(np.isfinite(row['loss_nats_per_dim']) and row['gradient_norm'] > 0 for row in rows)
    assert rows[0]['multiplier_mean'] == pytest.approx(1.0, abs=1e-4), 'training starts at lambda = 1'
    assert json.loads((trained / 'config.json').read_text()) == X.training_identity(tiny_config())
    complete = json.loads((trained / 'COMPLETE.train.json').read_text())
    assert complete['update'] == 6 and complete['provenance']['score_sha256'] == SCORE
    assert complete['provenance']['grid'] == list(GRID) and complete['provenance']['class_counts'] == [12, 8, 5, 2]
    payload = torch.load(trained / 'latest.pt', weights_only=False)
    assert payload['update'] == 6 and len(payload['rows']) == 6 and payload['ema'] is not None
    assert torch.load(trained / 'snapshot_3.pt', weights_only=False)['update'] == 3


def test_panel_bound_estimates_the_sum_of_edge_kls():
    """panel_bound against a fresh-noise Monte Carlo over all edges (an estimate of the same expectation)."""
    from indep_helpers import np_gaussian_kl, np_posterior_bayes
    config = tiny_config()
    schedule = Schedule.from_config(config)
    loss = X.build_loss(config['loss'], schedule, GRID)
    images, _, _ = tiny_dataset()
    labels = np.repeat(np.arange(4), [12, 8, 5, 2])
    panel = (np.tile(images, (150, 1, 1, 1)), np.tile(labels, 150))
    for lam in (0.5, 1.0):
        got = X.panel_bound(loss, FakeEpsilon(), ScaffoldKernel(schedule, FixedScaffold(lam)), panel, seed=1,
                            batch_size=512, device=CPU)
        rng = np.random.default_rng(0)
        x0 = np.tile(images.astype(np.float64).transpose(0, 3, 1, 2) / 127.5 - 1, (40, 1, 1, 1))
        y = np.tile(labels, 40)
        total = np.zeros(len(x0))
        for t, s in zip(GRID[:-2], GRID[1:-1]):
            ab_t, ab_s = TINY_AB[t], TINY_AB[s]
            x_t = np.sqrt(ab_t) * x0 + np.sqrt(1 - ab_t) * rng.normal(size=x0.shape)
            _, mean_p, variance_p, _ = np_edge(x_t, np_fake_epsilon(x_t, np.full(len(x0), t), y), lam, ab_t, ab_s)
            mean_q, variance_q = np_posterior_bayes(x0, x_t, ab_t, ab_s)
            total += np_gaussian_kl(mean_q, variance_q, mean_p, variance_p).reshape(len(x0), -1).mean(1)
        assert got['Overall'] == pytest.approx(total.mean(), rel=0.05), lam
        assert got['Many'] == got['Overall'], 'all four toy classes fall in the Many group'


def test_lambda1_reference_is_the_panel_bound_of_a_fixed_lambda1_kernel(trained):
    config = tiny_config()
    schedule = Schedule.from_config(config)
    loss = X.build_loss(config['loss'], schedule, GRID)
    images, indices, _ = tiny_dataset()
    panel = X.validation_panels(config, images, indices, None)['train']
    expected = X.panel_bound(loss, FakeEpsilon(), ScaffoldKernel(schedule, FixedScaffold(1.0)), panel, seed=5,
                             batch_size=7, device=CPU)
    assert json.loads((trained / 'reference_lambda1.json').read_text()) == {'train': expected}


def test_train_refuses_to_overwrite_or_resume_blindly(trained, tmp_path):
    with pytest.raises(FileExistsError):
        run_train(tiny_config(), trained)
    with pytest.raises(FileNotFoundError):
        run_train(tiny_config(), tmp_path / 'empty', resume=True)


@pytest.mark.parametrize('change', ['config', 'score', 'data'])
def test_resume_refuses_a_changed_config_score_or_data(trained, tmp_path, change):
    import shutil
    copy = tmp_path / 'run'
    shutil.copytree(trained, copy)
    config, arguments = tiny_config(), {}
    if change == 'config':
        config['training']['learning_rate'] = 5e-3
    elif change == 'score':
        arguments['epsilon'] = (FakeEpsilon(), 'another-score')
    else:
        arguments['data'] = tiny_dataset(seed=1)
    with pytest.raises(RuntimeError, match='different'):
        run_train(config, copy, resume=True, **arguments)


def test_resume_ignores_the_evaluation_block(trained, tmp_path):
    import shutil
    copy = tmp_path / 'run'
    shutil.copytree(trained, copy)
    config = tiny_config()
    config['evaluation']['seeds'] = [9]
    run_train(config, copy, resume=True)


class CrashAfterCheckpoint:
    """Fake score that dies on the first call made after ``latest.pt`` exists."""

    def __init__(self, checkpoint: Path):
        self.inner, self.checkpoint = FakeEpsilon(), checkpoint

    def __call__(self, x, t, labels):
        if self.checkpoint.is_file():
            raise KeyboardInterrupt('simulated preemption')
        return self.inner(x, t, labels)


def test_an_interrupted_run_resumes_bit_for_bit(trained, tmp_path):
    output = tmp_path / 'interrupted'
    with pytest.raises(KeyboardInterrupt):
        run_train(tiny_config(), output, epsilon=(CrashAfterCheckpoint(output / 'latest.pt'), SCORE))
    assert torch.load(output / 'latest.pt', weights_only=False)['update'] == 3
    assert not (output / 'COMPLETE.train.json').exists()
    run_train(tiny_config(), output, resume=True)
    straight = torch.load(trained / 'latest.pt', weights_only=False)
    resumed = torch.load(output / 'latest.pt', weights_only=False)
    for key in ('scaffold', 'ema'):
        assert all(torch.equal(straight[key][name], resumed[key][name]) for name in straight[key]), key
    assert straight['rows'] == resumed['rows'] and straight['update'] == resumed['update'] == 6
    assert (trained / 'train.jsonl').read_text() == (output / 'train.jsonl').read_text()
    assert straight['numpy_rng'] == resumed['numpy_rng'] and torch.equal(straight['noise_rng'], resumed['noise_rng'])


def test_training_is_reproducible_and_seed_sensitive(trained, tmp_path):
    run_train(tiny_config(), tmp_path / 'same')
    assert (trained / 'train.jsonl').read_text() == (tmp_path / 'same' / 'train.jsonl').read_text()
    config = tiny_config()
    config['training']['seed'] = 4
    run_train(config, tmp_path / 'other')
    assert (trained / 'train.jsonl').read_text() != (tmp_path / 'other' / 'train.jsonl').read_text()


def test_checkpoint_reload_gives_ema_and_online_weights(trained):
    payload = torch.load(trained / 'latest.pt', weights_only=False)
    ema, loaded = X.load_scaffold(trained / 'latest.pt', CPU)
    online, _ = X.load_scaffold(trained / 'latest.pt', CPU, weights='online')
    assert isinstance(ema, MLPScaffold) and not ema.training and not any(p.requires_grad for p in ema.parameters())
    assert loaded['update'] == 6
    assert all(torch.equal(ema.state_dict()[k], payload['ema'][k]) for k in payload['ema'])
    assert all(torch.equal(online.state_dict()[k], payload['scaffold'][k]) for k in payload['scaffold'])
    assert not torch.equal(ema.output.bias, online.output.bias)
    # EMA of a parameter that started at zero: |ema| <= max |online| along the run, and lags the last value.
    assert float(ema.output.bias.abs().max()) < float(online.output.bias.abs().max())
    with pytest.raises(ValueError):
        X.load_scaffold(trained / 'latest.pt', CPU, weights='best')


def test_load_scaffold_rejects_foreign_files_and_missing_ema(tmp_path):
    torch.save({'schema': 'something_else'}, tmp_path / 'foreign.pt')
    with pytest.raises(RuntimeError):
        X.load_scaffold(tmp_path / 'foreign.pt', CPU)
    config = tiny_config()
    config['training'].update(ema_decay=0.0, updates=2, checkpoint_interval=2, retain_snapshots=False)
    output = run_train(config, tmp_path / 'no_ema')
    assert sorted(p.name for p in output.glob('*.pt')) == ['latest.pt']
    with pytest.raises(RuntimeError, match='EMA'):
        X.load_scaffold(output / 'latest.pt', CPU)
    X.load_scaffold(output / 'latest.pt', CPU, weights='online')


@pytest.fixture(scope='module')
def learned(tmp_path_factory):
    """300 updates against a deliberately wrong score: phi has something to learn."""
    config = tiny_config()
    config['training'].update(updates=300, checkpoint_interval=100, retain_snapshots=False, ema_decay=0.0)
    return run_train(config, tmp_path_factory.mktemp('learn') / 'run'), config


def test_training_never_ends_worse_than_the_matched_lambda1_reference(learned):
    """Same panel, same (image, edge, noise) triples: phi starts at lambda = 1 and may only improve on it."""
    output, _ = learned
    reference = json.loads((output / 'reference_lambda1.json').read_text())['train']['Overall']
    rows = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
    bounds = [row['validation']['train']['Overall'] for row in rows if 'validation' in row]
    print(f'wrong-score toy: lambda=1 reference {reference:.4f}; phi after 100/200/300 updates {bounds}')
    assert bounds[-1] < reference and bounds[-1] <= bounds[0]
    assert all(np.isfinite(row['loss_nats_per_dim']) for row in rows)


def test_training_recovers_a_known_optimum_on_the_gaussian_control(tmp_path):
    """Positive control for the optimiser path.

    Pixels are i.i.d. N(0, 0.31^2) and the score is exact, so the best multiplier is close to 1.
    Starting from lambda = 0.3 (0.57 nats/dim per edge too high), training must close most of the gap
    to the lambda = 1 reference, measured on identical (image, edge, noise) triples.
    """
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(4), 40)
    images = np.clip(np.rint(127.5 + 40.0 * rng.normal(size=(160, 8, 8, 3))), 0, 255).astype(np.uint8)
    data = (images, [np.flatnonzero(labels == c) for c in range(4)], np.full(4, 40, dtype=np.float32))
    alpha, variance = torch.from_numpy(TINY_AB), (40.0 / 127.5) ** 2

    def exact_epsilon(x, t, y):
        a = alpha[t].to(x).reshape(-1, 1, 1, 1)
        return (1 - a).sqrt() * x / (a * variance + 1 - a)

    config = tiny_config()
    config['scaffold']['variance_initial'] = 0.3
    config['training'].update(updates=400, checkpoint_interval=400, retain_snapshots=False, ema_decay=0.0,
                              classes_per_batch=4, samples_per_class=8)
    config['validation']['per_class'] = 40
    output = run_train(config, tmp_path / 'run', epsilon=(exact_epsilon, SCORE), data=data)

    schedule = Schedule.from_config(config)
    loss = X.build_loss(config['loss'], schedule, GRID)
    panel = X.validation_panels(config, *data[:2], None)['train']
    bound = lambda scaffold: X.panel_bound(loss, exact_epsilon, ScaffoldKernel(schedule, scaffold), panel,
                                           seed=5, batch_size=7, device=CPU)['Overall']
    start, reference = bound(FixedScaffold(0.3)), bound(FixedScaffold(1.0))
    rows = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
    final = rows[-1]['validation']['train']['Overall']
    scaffold, _ = X.load_scaffold(output / 'latest.pt', CPU, weights='online')
    assert final == pytest.approx(bound(scaffold), rel=1e-6), 'the logged bound is the bound of the saved weights'
    print(f'gaussian control: lambda=0.3 start {start:.4f}, lambda=1 reference {reference:.4f}, trained {final:.4f}, '
          f'multiplier mean {rows[0]["multiplier_mean"]:.3f} -> {rows[-1]["multiplier_mean"]:.3f}')
    assert reference == pytest.approx(json.loads((output / 'reference_lambda1.json').read_text())['train']['Overall'])
    assert start - reference > 1.0, 'the control must have a gap to close'
    assert final - reference < 0.2 * (start - reference)
    assert rows[0]['multiplier_mean'] == pytest.approx(0.3, abs=0.01) and 0.6 < rows[-1]['multiplier_mean'] < 1.6


# Evaluation

def np_reference_samples(config, seed, *, lam=None, ddpm=None):
    """The whole evaluation pipeline for one fixed arm, rebuilt from PROTOCOL.json's wording."""
    settings = config['evaluation']
    labels = np.arange(settings['num_images']) % config['dataset']['num_classes']
    size, output = config['dataset']['image_size'], []
    for batch, start in enumerate(range(0, len(labels), settings['batch_size'])):
        y = labels[start:start + settings['batch_size']]
        shape = (len(y), 3, size, size)
        state = torch.randn(shape, generator=protocol_generator(seed, 0, batch)).double().numpy()
        brownian = protocol_generator(seed, 1, batch)
        for t, s in zip(GRID, GRID[1:]):
            ab_t, ab_s = TINY_AB[t], (1.0 if s == CLEAN else TINY_AB[s])
            eps = np_fake_epsilon(state, np.full(len(y), t), y)
            _, mean, variance, _ = np_edge(state, eps, 1.0 if ddpm else lam, ab_t, ab_s)
            if ddpm == 'fixedlarge':
                variance = 1 - ab_t / ab_s
            if s == CLEAN:
                state = mean
            else:
                state = mean + np.sqrt(variance) * torch.randn(shape, generator=brownian).double().numpy()
        output.append(((np.clip(state, -1, 1) + 1) / 2).transpose(0, 2, 3, 1))
    return np.concatenate(output)


@pytest.fixture(scope='module')
def evaluated(tmp_path_factory):
    output = tmp_path_factory.mktemp('evaluate') / 'panel'
    calls = []
    epsilon = FakeEpsilon()
    run_evaluate(tiny_config(), output, epsilon=(epsilon, SCORE), metric_fn=mean_metric(calls))
    return output, calls, epsilon


def test_evaluate_writes_the_documented_outputs(evaluated):
    output, calls, epsilon = evaluated
    assert {path.name for path in output.iterdir()} == {'PROTOCOL.json', 'summary.json', 'seed_0', 'seed_1'}
    arms = ['ddim', 'half', 'eta1', 'small', 'large']
    for seed in (0, 1):
        assert {path.name for path in (output / f'seed_{seed}').iterdir()} == set(arms)
        for arm in arms:
            directory = output / f'seed_{seed}' / arm
            assert {p.name for p in directory.iterdir()} == {'ARM.json', 'batches', 'variance', 'variance.json',
                                                             'samples.npy', 'metrics.json'}
            samples = np.load(directory / 'samples.npy')
            assert samples.shape == (8, 8, 8, 3) and samples.dtype == np.float32
            assert samples.min() >= 0 and samples.max() <= 1
            assert sorted(p.name for p in (directory / 'batches').iterdir()) == [f'batch_{i:06d}.npy' for i in range(3)]
    assert len(calls) == 10
    protocol = json.loads((output / 'PROTOCOL.json').read_text())
    assert protocol['grid'] == list(GRID) and protocol['nfe'] == 5 and protocol['score_sha256'] == SCORE
    assert epsilon.timesteps == list(GRID[:-1]) * (len(epsilon.timesteps) // 5), 'exactly K evaluations per batch'
    assert len(epsilon.timesteps) == 5 * 3 * 5 * 2  # nfe x batches x arms x seeds


@pytest.mark.parametrize('arm,oracle', [('ddim', {'lam': 0.0}), ('half', {'lam': 0.5}), ('eta1', {'lam': 1.0}),
                                        ('small', {'ddpm': 'fixedsmall'}), ('large', {'ddpm': 'fixedlarge'})])
@pytest.mark.parametrize('seed', [0, 1])
def test_stored_samples_equal_an_independent_numpy_pipeline(evaluated, arm, oracle, seed):
    """Latents, Brownian draws, cyclic labels, the kernel, the clamp and the NHWC layout, all at once."""
    samples = np.load(evaluated[0] / f'seed_{seed}' / arm / 'samples.npy')
    np.testing.assert_allclose(samples, np_reference_samples(tiny_config(), seed, **oracle), atol=2e-4)


def test_arms_are_paired_and_seeds_are_independent(evaluated):
    load = lambda seed, arm: np.load(evaluated[0] / f'seed_{seed}' / arm / 'samples.npy')
    np.testing.assert_allclose(load(0, 'eta1'), load(0, 'small'), atol=1e-6)  # the same kernel under two names
    assert np.abs(load(0, 'ddim') - load(1, 'ddim')).mean() > 0.01
    assert np.abs(load(0, 'ddim') - load(0, 'large')).mean() > 1e-3
    assert np.abs(load(0, 'ddim')[0] - load(0, 'ddim')[4]).mean() > 1e-3, 'same class, different latent'


def test_summary_and_reports_bind_metrics_to_samples(evaluated):
    output = evaluated[0]
    summary = json.loads((output / 'summary.json').read_text())
    assert set(summary['per_seed']) == {'ddim', 'half', 'eta1', 'small', 'large'}
    values = []
    for seed in (0, 1):
        report = json.loads((output / f'seed_{seed}' / 'ddim' / 'metrics.json').read_text())
        samples = output / f'seed_{seed}' / 'ddim' / 'samples.npy'
        assert report['samples_sha256'] == X.sha256_file(samples) and report['binding']['seed'] == seed
        assert report['metrics']['count'] == 8
        assert report['metrics']['FID']['Overall'] == pytest.approx(float(np.load(samples).mean()))
        assert summary['per_seed']['ddim'][str(seed)] == report['metrics']
        values.append(report['metrics']['FID']['Overall'])
    across = summary['across_seeds']['ddim']['FID.Overall']
    assert across['mean'] == pytest.approx(np.mean(values)) and across['std'] == pytest.approx(np.std(values, ddof=1))
    assert across['seeds'] == 2


def copy_panel(evaluated, tmp_path):
    import shutil
    shutil.copytree(evaluated[0], tmp_path / 'panel')
    return tmp_path / 'panel'


def test_rerun_recomputes_nothing(evaluated, tmp_path):
    output, calls, epsilon = copy_panel(evaluated, tmp_path), [], FakeEpsilon()
    before = {path: path.read_bytes() for path in output.rglob('*') if path.is_file()}
    run_evaluate(tiny_config(), output, epsilon=(epsilon, SCORE), metric_fn=mean_metric(calls))
    assert calls == [] and epsilon.timesteps == []
    assert {path: path.read_bytes() for path in output.rglob('*') if path.is_file()} == before


@pytest.mark.parametrize('damage', ['delete_shard', 'truncate_shard', 'wrong_shape_shard', 'delete_assembly'])
def test_lost_or_damaged_shards_are_regenerated_identically(evaluated, tmp_path, damage):
    output = copy_panel(evaluated, tmp_path)
    directory = output / 'seed_1' / 'half'
    original = np.load(directory / 'samples.npy')
    shard = directory / 'batches' / 'batch_000001.npy'
    (directory / 'samples.npy').unlink()
    if damage == 'delete_shard':
        shard.unlink()
    elif damage == 'truncate_shard':
        shard.write_bytes(shard.read_bytes()[:150])
    elif damage == 'wrong_shape_shard':
        np.save(shard, np.zeros((2, 8, 8, 3), dtype=np.float32))
    calls, epsilon = [], FakeEpsilon()
    run_evaluate(tiny_config(), output, epsilon=(epsilon, SCORE), metric_fn=mean_metric(calls))
    np.testing.assert_array_equal(np.load(directory / 'samples.npy'), original)
    assert len(epsilon.timesteps) == (0 if damage == 'delete_assembly' else 5), 'only the lost batch is resampled'
    assert calls == [], 'identical samples keep their metrics'


def test_stale_metrics_are_recomputed_when_samples_change(evaluated, tmp_path):
    output = copy_panel(evaluated, tmp_path)
    samples = output / 'seed_0' / 'ddim' / 'samples.npy'
    np.save(samples, np.zeros((8, 8, 8, 3), dtype=np.float32))
    calls = []
    run_evaluate(tiny_config(), output, metric_fn=mean_metric(calls))
    assert calls == [samples]
    assert json.loads((samples.parent / 'metrics.json').read_text())['metrics']['FID']['Overall'] == 0.0


def test_a_changed_protocol_or_arm_definition_is_refused(evaluated, tmp_path):
    output = copy_panel(evaluated, tmp_path)
    for mutate in (lambda c: c['sampler']['grid'].update(nfe=4), lambda c: c['evaluation'].update(num_images=12),
                   lambda c: c['sampler'].update(guidance_scale=1.0), lambda c: c['evaluation'].update(batch_size=4)):
        config = tiny_config()
        mutate(config)
        with pytest.raises(RuntimeError, match='different protocol'):
            run_evaluate(config, output)
    with pytest.raises(RuntimeError, match='different protocol'):
        run_evaluate(tiny_config(), output, epsilon=(FakeEpsilon(), 'another-score'))
    config = tiny_config()
    config['evaluation']['arms'][1]['lambda'] = 0.75
    with pytest.raises(RuntimeError, match='different arm'):
        run_evaluate(config, output)
    more_seeds = tiny_config()
    more_seeds['evaluation']['seeds'] = [0, 1, 2]
    run_evaluate(more_seeds, output)
    assert (output / 'seed_2' / 'ddim' / 'samples.npy').is_file(), 'seeds can be added to a panel'


def test_evaluate_argument_validation(tmp_path):
    for mutate in (lambda c: c['evaluation'].update(num_images=10), lambda c: c['evaluation'].update(seeds=[0, 0]),
                   lambda c: c['evaluation'].update(seeds=[]),
                   lambda c: c['evaluation']['arms'].append({'name': 'ddim', 'kind': 'fixed', 'lambda': 0.1})):
        config = tiny_config()
        mutate(config)
        with pytest.raises(ValueError):
            run_evaluate(config, tmp_path / 'bad')
    with pytest.raises(ValueError, match='unknown arms'):
        run_evaluate(tiny_config(), tmp_path / 'bad', arms=['nope'])
    config = tiny_config()
    config['evaluation']['arms'] = [{'name': 'x', 'kind': 'heun'}]
    with pytest.raises(ValueError):
        run_evaluate(config, tmp_path / 'bad2')


def test_sample_only_and_arm_selection(tmp_path):
    calls = []
    output = run_evaluate(tiny_config(), tmp_path / 'panel', arms=['half'], sample_only=True, metric_fn=mean_metric(calls))
    assert calls == [] and not (output / 'summary.json').exists()
    assert [p.name for p in (output / 'seed_0').iterdir()] == ['half']
    assert not (output / 'seed_0' / 'half' / 'metrics.json').exists()
    run_evaluate(tiny_config(), output, arms=['half', 'ddim'], metric_fn=mean_metric(calls))
    assert len(calls) == 4
    assert set(json.loads((output / 'summary.json').read_text())['per_seed']) == {'ddim', 'half'}


# Learned arms

def learned_config(checkpoint, **arm):
    config = tiny_config()
    config['evaluation']['arms'] = [{'name': 'ddim', 'kind': 'fixed', 'lambda': 0.0},
                                    {'name': 'phi', 'kind': 'learned', 'checkpoint': str(checkpoint), **arm}]
    return config


def test_learned_arm_samples_with_the_trained_scaffold(learned, tmp_path):
    run, _ = learned
    checkpoint = run / 'latest.pt'
    output = run_evaluate(learned_config(checkpoint, weights='online'), tmp_path / 'panel')
    binding = json.loads((output / 'seed_0' / 'phi' / 'ARM.json').read_text())['arm']
    assert binding['checkpoint_sha256'] == X.sha256_file(checkpoint) and binding['update'] == 300
    assert binding['weights'] == 'online' and binding['state'] == 'own' and binding['protocol_mismatch'] == []
    assert binding['trained_grid'] == list(GRID)

    # Rebuild the arm by hand: same latents and Brownian draws, lambda from the checkpointed weights.
    scaffold, _ = X.load_scaffold(checkpoint, CPU, weights='online')
    config = tiny_config()
    labels = np.arange(8) % 4
    expected = []
    for batch, start in enumerate(range(0, 8, 3)):
        y = labels[start:start + 3]
        state = torch.randn((len(y), 3, 8, 8), generator=protocol_generator(0, 0, batch)).double().numpy()
        brownian = protocol_generator(0, 1, batch)
        for t, s in zip(GRID, GRID[1:]):
            ab_t, ab_s = TINY_AB[t], (1.0 if s == CLEAN else TINY_AB[s])
            with torch.no_grad():
                lam = scaffold(torch.from_numpy(state).float(), torch.full((len(y),), (t + 1) / 50), torch.from_numpy(y))
            _, mean, variance, _ = np_edge(state, np_fake_epsilon(state, np.full(len(y), t), y), lam.double().numpy(), ab_t, ab_s)
            state = mean if s == CLEAN else mean + np.sqrt(variance) * torch.randn(state.shape, generator=brownian).double().numpy()
        expected.append(((np.clip(state, -1, 1) + 1) / 2).transpose(0, 2, 3, 1))
    np.testing.assert_allclose(np.load(output / 'seed_0' / 'phi' / 'samples.npy'), np.concatenate(expected), atol=5e-4)
    assert config['diffusion']['steps'] == 50


def test_learned_arm_protocol_guards(learned, tmp_path):
    checkpoint = learned[0] / 'latest.pt'
    with pytest.raises(RuntimeError, match='EMA'):
        run_evaluate(learned_config(checkpoint), tmp_path / 'ema')  # trained with ema_decay = 0
    other_grid = learned_config(checkpoint, weights='online')
    other_grid['sampler']['grid']['nfe'] = 10
    with pytest.raises(RuntimeError, match='grid'):
        run_evaluate(other_grid, tmp_path / 'grid')
    waived = learned_config(checkpoint, weights='online', allow_protocol_mismatch=True)
    waived['sampler']['grid']['nfe'] = 10
    output = run_evaluate(waived, tmp_path / 'waived')
    assert json.loads((output / 'seed_0' / 'phi' / 'ARM.json').read_text())['arm']['protocol_mismatch'] == ['grid']
    for arm in ({'weights': 'online'}, {'weights': 'online', 'allow_protocol_mismatch': True}):
        with pytest.raises(RuntimeError, match='score'):
            run_evaluate(learned_config(checkpoint, **arm), tmp_path / 'score', epsilon=(FakeEpsilon(), 'another-score'))
    schedule = learned_config(checkpoint, weights='online', allow_protocol_mismatch=True)
    schedule['diffusion']['beta_T'] = 0.1
    with pytest.raises(RuntimeError, match='diffusion'):
        run_evaluate(schedule, tmp_path / 'schedule')
    with pytest.raises(ValueError):
        run_evaluate(learned_config(checkpoint, weights='online', state='random'), tmp_path / 'state')


def test_shuffled_state_arm_differs_and_needs_two_samples_per_batch(learned, tmp_path):
    checkpoint = learned[0] / 'latest.pt'
    config = learned_config(checkpoint, weights='online')
    config['evaluation']['arms'].append({'name': 'shuffled', 'kind': 'learned', 'checkpoint': str(checkpoint),
                                         'weights': 'online', 'state': 'shuffled'})
    config['evaluation'].update(batch_size=8)  # 4 classes, cyclic labels -> two samples of every class per batch
    output = run_evaluate(config, tmp_path / 'panel')
    own, shuffled = (np.load(output / 'seed_0' / name / 'samples.npy') for name in ('phi', 'shuffled'))
    assert own.shape == shuffled.shape and np.abs(own - shuffled).max() > 1e-5
    lonely = learned_config(checkpoint, weights='online', state='shuffled')
    lonely['evaluation'].update(batch_size=7)  # 8 images -> a final batch of one
    with pytest.raises(ValueError, match='two samples'):
        run_evaluate(lonely, tmp_path / 'lonely')


def test_time_only_training_yields_a_state_blind_scaffold(tmp_path):
    config = tiny_config()
    config['scaffold']['inputs'] = 'time_class'
    config['training'].update(updates=4, checkpoint_interval=4, retain_snapshots=False)
    output = run_train(config, tmp_path / 'time_only')
    scaffold, _ = X.load_scaffold(output / 'latest.pt', CPU)
    labels, time = torch.tensor([0, 1, 2, 3]), torch.tensor([0.1, 0.4, 0.7, 1.0])
    first = scaffold(torch.randn(4, 3, 8, 8), time, labels)
    assert torch.equal(first, scaffold(torch.randn(4, 3, 8, 8) * 5, time, labels))
    assert float((first - 1.0).abs().max()) > 1e-4, 'four updates moved lambda away from its initial value'
