"""Gate C: training resume and evaluation reproducibility, on CPU with a stub denoiser."""
from __future__ import annotations

import copy
import json
import shutil

import numpy as np
import pytest
import torch

from scaffold import main as runtime
from scaffold.backbone import FixedScaffold
from scaffold.sampling import ScaffoldKernel, Schedule, build_grid, sample
from scaffold.utils import stream_generator

CLASSES, SIZE = 4, 8
DEVICE = torch.device('cpu')


def epsilon_fn(x, t, labels):
    return 0.3 * x + 0.05 * labels[:, None, None, None]


EPSILON = (epsilon_fn, 'stub-score')


def make_config(checkpoint='unused', **training):
    return {
        'paths': {'checkpoint': 'unused', 'checkpoint_sha256': 'stub'},
        'dataset': {'num_classes': CLASSES, 'image_size': SIZE},
        'diffusion': {'steps': 1000, 'beta_1': 1e-4, 'beta_T': 0.02},
        'model': {'microbatch': 16},
        'sampler': {'grid': {'nfe': 5, 'spacing': 'leading'}, 'guidance_scale': 0.0},
        'scaffold': {'kind': 'mlp', 'inputs': 'state_time_class', 'image_size': SIZE, 'hidden_dims': [16, 16],
                     'condition_dim': 8, 'variance_floor': 1e-4, 'variance_initial': 1.0},
        'loss': {'kind': 'respaced_vlb'},
        'training': {'name': 'unit', 'seed': 123, 'classes_per_batch': 2, 'samples_per_class': 4, 'updates': 6,
                     'learning_rate': 1e-2, 'weight_decay': 0.0, 'gradient_clip': 1.0, 'horizontal_flip': True,
                     'ema_decay': 0.9, 'checkpoint_interval': 3, 'log_interval': 1, 'retain_snapshots': True,
                     **training},
        'validation': {'per_class': 3, 'seed': 7, 'batch_size': 8, 'heldout': False},
        'evaluation': {
            'name': 'unit_eval', 'num_images': 16, 'batch_size': 8, 'seeds': [0, 1],
            'arms': [{'name': 'ddim', 'kind': 'fixed', 'lambda': 0.0},
                     {'name': 'eta1', 'kind': 'fixed', 'lambda': 1.0},
                     {'name': 'ddpm_large', 'kind': 'ddpm', 'variance_type': 'fixedlarge'},
                     {'name': 'learned', 'kind': 'learned', 'checkpoint': str(checkpoint)},
                     {'name': 'shuffled', 'kind': 'learned', 'checkpoint': str(checkpoint), 'state': 'shuffled'}]},
    }


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    counts = np.array([12, 8, 5, 4])
    images = rng.integers(0, 256, size=(int(counts.sum()), SIZE, SIZE, 3), dtype=np.uint8)
    labels = np.repeat(np.arange(CLASSES), counts)
    return images, [np.flatnonzero(labels == c) for c in range(CLASSES)], counts.astype(np.float32)


@pytest.fixture(scope='module')
def trained(tmp_path_factory, data):
    output = tmp_path_factory.mktemp('train') / 'run'
    runtime.train(make_config(), DEVICE, output=output, epsilon=EPSILON, data=data)
    return output


def metric_fn(samples, labels):
    return {'mean_pixel': float(np.load(samples).mean()), 'count': len(labels)}


def run_eval(config, output, **options):
    return runtime.evaluate(config, DEVICE, output=output, epsilon=EPSILON, metric_fn=metric_fn, **options)


def test_training_writes_validated_checkpoints(trained):
    payload = torch.load(trained / 'latest.pt', weights_only=False)
    assert payload['update'] == 6 and payload['ema'] is not None
    rows = [json.loads(line) for line in (trained / 'train.jsonl').read_text().splitlines()]
    assert [row['update'] for row in rows] == list(range(1, 7))
    assert 'validation' not in rows[1] and np.isfinite(rows[2]['validation']['train']['Overall'])
    assert (trained / 'reference_lambda1.json').is_file() and (trained / 'COMPLETE.train.json').is_file()
    scaffold, _ = runtime.load_scaffold(trained / 'latest.pt', DEVICE, weights='ema')
    assert not any(p.requires_grad for p in scaffold.parameters())
    assert 'evaluation' not in payload['config']


def test_resume_reproduces_uninterrupted_training(trained, tmp_path, data):
    resumed = tmp_path / 'resumed'
    resumed.mkdir()
    shutil.copy(trained / 'snapshot_3.pt', resumed / 'latest.pt')
    runtime.train(make_config(), DEVICE, resume=True, output=resumed, epsilon=EPSILON, data=data)
    full = torch.load(trained / 'latest.pt', weights_only=False)
    again = torch.load(resumed / 'latest.pt', weights_only=False)
    for key in ('scaffold', 'ema'):
        assert all(torch.equal(full[key][name], again[key][name]) for name in full[key])
    assert full['rows'] == again['rows']


def test_resume_refuses_a_changed_config(trained, tmp_path, data):
    resumed = tmp_path / 'changed'
    resumed.mkdir()
    shutil.copy(trained / 'snapshot_3.pt', resumed / 'latest.pt')
    with pytest.raises(RuntimeError):
        runtime.train(make_config(learning_rate=1e-3), DEVICE, resume=True, output=resumed, epsilon=EPSILON, data=data)
    with pytest.raises(FileExistsError):
        runtime.train(make_config(), DEVICE, output=trained, epsilon=EPSILON, data=data)


def test_evaluation_settings_are_not_part_of_a_checkpoints_identity(trained, tmp_path, data):
    """Changing seeds, sample counts or arms must not invalidate a trained scaffold."""
    resumed = tmp_path / 'same'
    resumed.mkdir()
    shutil.copy(trained / 'snapshot_3.pt', resumed / 'latest.pt')
    config = make_config()
    config['evaluation'].update(num_images=400, seeds=[5, 6, 7], arms=[])
    runtime.train(config, DEVICE, resume=True, output=resumed, epsilon=EPSILON, data=data)


def test_noise_streams_are_independent():
    draw = lambda *key: torch.randn(8, generator=stream_generator(DEVICE, *key))
    assert torch.equal(draw(0, 0, 0), draw(0, 0, 0))
    cells = [draw(seed, stream, batch) for seed in range(3) for stream in range(2) for batch in range(3)]
    assert len({tuple(cell.tolist()) for cell in cells}) == len(cells)


def test_evaluation_is_paired_seeded_and_resumable(trained, tmp_path):
    config = make_config(trained / 'latest.pt')
    output = run_eval(config, tmp_path / 'eval')
    load = lambda seed, arm: np.load(output / f'seed_{seed}' / arm / 'samples.npy')

    # Seeds are independent: no batch of seed 1 repeats a batch of seed 0 (the old seed+batch bug).
    first, second = load(0, 'eta1'), load(1, 'eta1')
    assert first.shape == (16, SIZE, SIZE, 3) and first.min() >= 0 and first.max() <= 1
    assert not any(np.array_equal(first[a:a + 8], second[b:b + 8]) for a in (0, 8) for b in (0, 8))

    # The DDIM arm is exactly a direct call of the shared sampler with the protocol's streams.
    schedule = Schedule.from_config(config)
    labels = torch.arange(8) % CLASSES
    initial = torch.randn((8, 3, SIZE, SIZE), generator=stream_generator(DEVICE, 0, runtime.LATENT_STREAM, 0))
    direct = sample(epsilon_fn, ScaffoldKernel(schedule, FixedScaffold(0.0)), labels, initial,
                    build_grid(1000, config['sampler']['grid']),
                    generator=stream_generator(DEVICE, 0, runtime.BROWNIAN_STREAM, 0))
    assert np.array_equal(load(0, 'ddim')[:8], runtime.to_unit_nhwc(direct))
    assert not np.array_equal(load(0, 'learned'), load(0, 'shuffled'))

    summary = json.loads((output / 'summary.json').read_text())
    assert summary['across_seeds']['ddim']['mean_pixel']['seeds'] == 2

    # Losing files mid-run regenerates identical samples instead of leaving zeros behind.
    reference = load(0, 'learned').copy()
    arm = output / 'seed_0' / 'learned'
    (arm / 'samples.npy').unlink()
    (arm / 'metrics.json').unlink()
    (arm / 'batches' / 'batch_000000.npy').unlink()
    (arm / 'batches' / 'batch_000001.npy').write_bytes(b'truncated')
    run_eval(config, output)
    assert np.array_equal(load(0, 'learned'), reference)


def test_evaluation_refuses_protocol_or_arm_drift(trained, tmp_path):
    config = make_config(trained / 'latest.pt')
    output = run_eval(config, tmp_path / 'eval', arms=['ddim', 'eta1'])
    assert not (output / 'seed_0' / 'learned').exists()
    changed = copy.deepcopy(config)
    changed['sampler']['guidance_scale'] = 1.5
    with pytest.raises(RuntimeError, match='different protocol'):
        run_eval(changed, output, arms=['ddim'])
    renamed = copy.deepcopy(config)
    renamed['evaluation']['arms'][1]['lambda'] = 0.5
    with pytest.raises(RuntimeError, match='different arm definition'):
        run_eval(renamed, output, arms=['eta1'])
    with pytest.raises(ValueError, match='unknown arms'):
        run_eval(config, output, arms=['missing'])
    # Adding an arm to an existing panel is fine and leaves finished arms untouched.
    extended = copy.deepcopy(config)
    extended['evaluation']['arms'].append({'name': 'half', 'kind': 'fixed', 'lambda': 0.5})
    stamp = (output / 'seed_0' / 'ddim' / 'samples.npy').stat().st_mtime_ns
    run_eval(extended, output, arms=['ddim', 'half'])
    assert (output / 'seed_0' / 'ddim' / 'samples.npy').stat().st_mtime_ns == stamp
    assert (output / 'seed_1' / 'half' / 'samples.npy').is_file()


def test_learned_arm_must_match_the_evaluation_protocol(trained, tmp_path):
    """A checkpoint trained for 5 NFE is not silently sampled at 10 NFE."""
    config = make_config(trained / 'latest.pt')
    config['sampler']['grid'] = {'nfe': 10, 'spacing': 'leading'}
    with pytest.raises(RuntimeError, match='grid'):
        run_eval(config, tmp_path / 'strict', arms=['learned'])
    config['evaluation']['arms'][3]['allow_protocol_mismatch'] = True
    output = run_eval(config, tmp_path / 'explicit', arms=['learned'])
    binding = json.loads((output / 'seed_0' / 'learned' / 'ARM.json').read_text())
    assert binding['arm']['protocol_mismatch'] == ['grid']
    with pytest.raises(RuntimeError, match='score'):
        runtime.evaluate(config, DEVICE, arms=['learned'], output=tmp_path / 'score',
                         epsilon=(epsilon_fn, 'another-score'), metric_fn=metric_fn)
