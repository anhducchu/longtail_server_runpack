"""utils.py: config overrides, paths, hashing, atomic output, seeding and the long-tailed split."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from indep_helpers import protocol_generator
from scaffold import utils as U
from scaffold.backbone import build_scaffold
from scaffold.loss import build_loss
from scaffold.sampling import Schedule, build_grid

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / 'scaffold' / 'config.json'
DATA = REPOSITORY / 'data' / 'cifar-100-python'


# Configuration

def test_shipped_config_loads_and_builds_every_component():
    config = U.load_config(CONFIG)
    assert set(U.CONFIG_SECTIONS) <= set(config)
    schedule = Schedule.from_config(config)
    grid = build_grid(len(schedule), config['sampler']['grid'])
    assert len(schedule) == 1000 and len(grid) == 21
    assert build_loss(config['loss'], schedule, grid).num_edges == 19
    scaffold = build_scaffold(config['scaffold'], config['dataset']['num_classes'])
    state = torch.zeros(2, 3, config['dataset']['image_size'], config['dataset']['image_size'])
    assert scaffold(state, 0.5, torch.tensor([0, 99])).shape == state.shape
    names = [arm['name'] for arm in config['evaluation']['arms']]
    assert len(names) == len(set(names))
    assert config['evaluation']['num_images'] % config['dataset']['num_classes'] == 0
    assert config['evaluation']['num_images'] % config['evaluation']['batch_size'] != 1, \
        'a final batch of one sample would crash the shuffled-state arm'


def test_overrides_parse_json_and_fall_back_to_strings():
    config = U.load_config(CONFIG, ['sampler.grid.nfe=50', 'sampler.guidance_scale=1.5',
                                    'scaffold.inputs=time_class', 'training.name=time_only_nfe20',
                                    'validation.heldout=false', 'evaluation.seeds=[7, 8]'])
    assert config['sampler']['grid'] == {'nfe': 50, 'spacing': 'leading'}
    assert config['sampler']['guidance_scale'] == 1.5 and config['scaffold']['inputs'] == 'time_class'
    assert config['training']['name'] == 'time_only_nfe20' and config['validation']['heldout'] is False
    assert config['evaluation']['seeds'] == [7, 8]
    replaced = U.load_config(CONFIG, ['sampler.grid={"stride": 10}'])
    assert replaced['sampler']['grid'] == {'stride': 10}
    assert U.load_config(CONFIG, ['training.name=a=b'])['training']['name'] == 'a=b'
    assert json.loads(CONFIG.read_text())['sampler']['grid']['nfe'] == 20, 'the file itself is untouched'


def test_bad_overrides_and_incomplete_configs_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        U.load_config(CONFIG, ['sampler.grid.nfe'])
    with pytest.raises(KeyError):
        U.load_config(CONFIG, ['nosuchsection.key=1'])
    partial = json.loads(CONFIG.read_text())
    del partial['loss']
    path = tmp_path / 'partial.json'
    path.write_text(json.dumps(partial))
    with pytest.raises(ValueError, match='loss'):
        U.load_config(path)


def test_paths_resolve_against_the_repository_root(tmp_path):
    assert U.ROOT == REPOSITORY and U.SCAFFOLD_ROOT == REPOSITORY / 'scaffold'
    assert U.rooted_path('scaffold/x') == REPOSITORY / 'scaffold' / 'x'
    assert U.rooted_path(str(tmp_path / 'a')) == (tmp_path / 'a').resolve()
    assert U.output_directory('learned_nfe20') == REPOSITORY / 'scaffold' / 'outputs' / 'learned_nfe20'


@pytest.mark.parametrize('name', ['', '../escape', 'a/b', '.hidden', 'with space', '/abs', 'x\n'])
def test_unsafe_run_names_are_rejected(name):
    with pytest.raises(ValueError):
        U.output_directory(name)


# Hashing

def test_file_and_array_hashes(tmp_path):
    payload = bytes(range(256)) * 1000
    path = tmp_path / 'blob.bin'
    path.write_bytes(payload)
    assert U.sha256_file(path) == hashlib.sha256(payload).hexdigest()
    values = np.arange(12, dtype=np.float32)
    assert U.array_sha256(values) == U.array_sha256(values.copy())
    assert U.array_sha256(values) == U.array_sha256(np.asfortranarray(values))
    distinct = {U.array_sha256(values), U.array_sha256(values.reshape(3, 4)), U.array_sha256(values.astype(np.float64)),
                U.array_sha256(values + 1)}
    assert len(distinct) == 4


def test_canonical_hash_ignores_key_order_only():
    assert U.canonical_hash({'a': 1, 'b': {'c': 2, 'd': [1, 2]}}) == U.canonical_hash({'b': {'d': [1, 2], 'c': 2}, 'a': 1})
    assert U.canonical_hash({'a': [1, 2]}) != U.canonical_hash({'a': [2, 1]})
    assert U.canonical_hash({'a': 1}) != U.canonical_hash({'a': 1.5})


# Atomic output

def test_atomic_writers_round_trip_and_leave_no_temporaries(tmp_path):
    U.atomic_json(tmp_path / 'deep' / 'a.json', {'b': 1, 'a': [1.5, None]})
    assert json.loads((tmp_path / 'deep' / 'a.json').read_text()) == {'a': [1.5, None], 'b': 1}
    array = np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
    U.atomic_numpy(tmp_path / 'a.npy', array)
    np.testing.assert_array_equal(np.load(tmp_path / 'a.npy'), array)
    U.atomic_npz(tmp_path / 'a.npz', x=array, tag=np.asarray('abc'))
    with np.load(tmp_path / 'a.npz') as loaded:
        np.testing.assert_array_equal(loaded['x'], array)
        assert str(loaded['tag']) == 'abc'
    U.atomic_torch_save(tmp_path / 'a.pt', {'w': torch.arange(3)})
    assert torch.load(tmp_path / 'a.pt')['w'].tolist() == [0, 1, 2]
    assert sorted(p.name for p in tmp_path.iterdir()) == ['a.npy', 'a.npz', 'a.pt', 'deep']
    assert [p.name for p in (tmp_path / 'deep').iterdir()] == ['a.json']


def test_a_failed_write_neither_publishes_nor_destroys(tmp_path):
    target = tmp_path / 'result.json'
    U.atomic_json(target, {'version': 1})
    with pytest.raises(RuntimeError):
        with U.atomic_output(target) as temporary:
            temporary.write_text('{"version": 2, "trunc')
            raise RuntimeError('crash mid-write')
    assert json.loads(target.read_text()) == {'version': 1}
    assert [p.name for p in tmp_path.iterdir()] == ['result.json']
    fresh = tmp_path / 'never.npy'
    with pytest.raises(TypeError):
        U.atomic_json(fresh, {'bad': {1, 2}} if False else object.__new__(type('Unserialisable', (), {'__str__': None})))
    assert not fresh.exists() and [p.name for p in tmp_path.iterdir()] == ['result.json']


# Randomness

def test_stream_generators_follow_the_documented_seedsequence_protocol():
    device = torch.device('cpu')
    draw = lambda *key: torch.randn(64, generator=U.stream_generator(device, *key))
    assert torch.equal(draw(3, 0, 5), draw(3, 0, 5))
    assert torch.equal(draw(3, 0, 5), torch.randn(64, generator=protocol_generator(3, 0, 5)))
    keys = [(seed, stream, batch) for seed in range(4) for stream in (0, 1) for batch in range(4)]
    draws = torch.stack([draw(*key) for key in keys])
    correlation = np.corrcoef(draws.numpy())
    np.fill_diagonal(correlation, 0.0)
    assert np.abs(correlation).max() < 0.6, 'streams of nearby keys must be unrelated'
    assert len({tuple(row.tolist()) for row in draws}) == len(keys)


def test_set_reproducible_seed_resets_all_three_generators():
    import random
    U.set_reproducible_seed(11)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    U.set_reproducible_seed(11)
    assert first == (random.random(), np.random.rand(), torch.rand(1).item())


# Data

def test_to_pixels_maps_uint8_nhwc_to_unit_range_nchw():
    images = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    images[0, 1, 2] = [0, 255, 51]
    pixels = U.to_pixels(images, torch.device('cpu'))
    assert pixels.shape == (2, 3, 4, 5) and pixels.dtype == torch.float32
    assert pixels[0, :, 1, 2].tolist() == pytest.approx([-1.0, 1.0, 51 / 127.5 - 1])
    assert float(pixels.min()) == -1.0 and float(pixels.max()) == 1.0


def test_long_tail_counts_follow_the_exponential_profile():
    """Synthetic stand-in for CIFAR-100: 100 classes x 500 images, no files needed."""
    dataset = object.__new__(U.ImbalanceCIFAR100)
    labels = np.tile(np.arange(100), 500)
    dataset.data = np.arange(50_000, dtype=np.int64).reshape(-1, 1)
    dataset.targets = labels.tolist()
    dataset.num_per_cls_dict = {}
    counts = dataset._image_counts('exp', 0.01)
    assert counts == [int(500 * 0.01 ** (i / 99)) for i in range(100)]
    assert counts[0] == 500 and counts[-1] == 5 and sum(counts) == 10847  # the published CIFAR-100-LT (IF=100) size
    assert dataset._image_counts('step', 0.1) == [500] * 50 + [50] * 50
    np.random.seed(0)
    dataset._make_imbalanced(counts)
    assert len(dataset.data) == 10847 and np.bincount(dataset.targets).tolist() == counts
    assert len(np.unique(dataset.data)) == 10847, 'no image may be drawn twice'
    assert (labels[dataset.data[:, 0]] == np.asarray(dataset.targets)).all(), 'images keep their own label'


@pytest.mark.real_assets
def test_real_long_tailed_split_is_deterministic_and_consistent():
    if not DATA.is_dir():
        pytest.skip('data/cifar-100-python is absent')
    config = U.load_config(CONFIG)
    images, indices, counts = U.load_training_data(config)
    again = U.load_training_data(config)[0]
    assert U.array_sha256(images) == U.array_sha256(again)
    assert images.shape == (10847, 32, 32, 3) and images.dtype == np.uint8
    assert counts.tolist() == [int(500 * 0.01 ** (i / 99)) for i in range(100)]
    assert [len(i) for i in indices] == counts.astype(int).tolist()

    # The frozen score was trained on the vendor's split: same images, same order.
    from importlib import import_module
    vendor = import_module('third_party.CCUA-DDPM.dataset').ImbalanceCIFAR100(
        root=str(REPOSITORY / 'data'), imb_type='exp', imb_factor=0.01, rand_number=0, train=True, download=False)
    np.testing.assert_array_equal(images, vendor.data)
    assert np.concatenate([np.full(len(i), c) for c, i in enumerate(indices)]).tolist() == list(vendor.targets)

    import torchvision
    full = torchvision.datasets.CIFAR100(str(REPOSITORY / 'data'), train=True, download=False)
    owner = {}
    for image, label in zip(full.data, full.targets):
        owner.setdefault(image.tobytes(), set()).add(label)
    for class_id in (0, 33, 66, 99):
        assert all(class_id in owner[image.tobytes()] for image in images[indices[class_id]])

    heldout_images, heldout_labels = U.load_heldout_data(config)
    assert heldout_images.shape == (10000, 32, 32, 3) and np.bincount(heldout_labels).tolist() == [100] * 100
    assert not any(image.tobytes() in owner for image in heldout_images[:500]), 'held-out images are not training images'
