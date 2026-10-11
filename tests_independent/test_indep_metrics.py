"""metrics.py against scipy / brute-force oracles, plus positive controls on the pinned assets."""
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg
import scipy.stats
import torch

from scaffold import metrics as M
from scaffold.utils import sha256_file

CPU = torch.device('cpu')
REPOSITORY = Path(__file__).resolve().parents[1]
STATS = REPOSITORY / 'third_party' / 'CCUA-DDPM' / 'stats'
DATA = REPOSITORY / 'data' / 'cifar-100-python'
HUB = Path(torch.hub.get_dir()) / 'checkpoints'


# Oracles

def scipy_fid(mean_a, covariance_a, mean_b, covariance_b):
    root = scipy.linalg.sqrtm(covariance_a @ covariance_b).real
    return float(((mean_a - mean_b) ** 2).sum() + np.trace(covariance_a) + np.trace(covariance_b) - 2 * np.trace(root))


def scipy_inception_score(probabilities, splits=10):
    scores = []
    count = len(probabilities)  # split boundaries i * N // splits, the pytorch-gan-metrics / vendor convention
    for block in (probabilities[i * count // splits:(i + 1) * count // splits] for i in range(splits)):
        marginal = block.mean(axis=0)
        scores.append(np.exp(np.mean([scipy.stats.entropy(row, marginal) for row in block])))
    return float(np.mean(scores)), float(np.std(scores))


def brute_distances(a, b):
    return np.sqrt(((a[:, None, :].astype(np.float64) - b[None, :, :].astype(np.float64)) ** 2).sum(-1))


def clouds(seed=0, generated=150, reference=220, dimensions=6):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(generated, dimensions)).astype(np.float32) * 1.2 + 0.4,
            rng.normal(size=(reference, dimensions)).astype(np.float32))


# FID

def test_groups_partition_the_hundred_classes():
    assert [len(M.GROUPS[name]) for name in ('Many', 'Medium', 'Few')] == [33, 33, 34]
    assert np.concatenate(list(M.GROUPS.values())).tolist() == list(range(100))


def test_moments_match_numpy():
    features = np.random.default_rng(0).normal(size=(50, 5)).astype(np.float32)
    mean, covariance = M._moments(features, CPU)
    assert mean.dtype == torch.float64
    np.testing.assert_allclose(mean.numpy(), features.astype(np.float64).mean(0), rtol=1e-12)
    np.testing.assert_allclose(covariance.numpy(), np.cov(features.astype(np.float64), rowvar=False), rtol=1e-10)
    for bad in (features[:1], features[0]):
        with pytest.raises(ValueError):
            M._moments(bad, CPU)


@pytest.mark.parametrize('dimensions,count', [(4, 200), (12, 40), (16, 10)])
def test_frechet_distance_matches_scipy_sqrtm(dimensions, count):
    """Includes rank-deficient covariances (count < dimensions), the regime of 2048-d group FID."""
    rng = np.random.default_rng(dimensions)
    a = rng.normal(size=(count, dimensions)) @ rng.normal(size=(dimensions, dimensions))
    b = rng.normal(size=(count + 5, dimensions)) * rng.uniform(0.5, 2, size=dimensions) + rng.normal(size=dimensions)
    moments_a, moments_b = M._moments(a, CPU), M._moments(b, CPU)
    expected = scipy_fid(*(value.numpy() for value in moments_a + moments_b))
    assert M.frechet_distance(*moments_a, *moments_b) == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert M.frechet_distance(*moments_b, *moments_a) == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert abs(M.frechet_distance(*moments_a, *moments_a)) < 1e-6 * float(torch.trace(moments_a[1]))


def test_frechet_distance_closed_forms():
    zero, identity = torch.zeros(3, dtype=torch.float64), torch.eye(3, dtype=torch.float64)
    shift = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float64)
    assert M.frechet_distance(zero, identity, shift, identity) == pytest.approx(9.0)
    # Diagonal Gaussians: sum (sqrt(a) - sqrt(b))^2.
    a, b = torch.diag(torch.tensor([4.0, 1.0, 9.0])).double(), torch.diag(torch.tensor([1.0, 1.0, 4.0])).double()
    assert M.frechet_distance(zero, a, zero, b) == pytest.approx(1.0 + 0.0 + 1.0)


# Inception score

def test_inception_score_limits_and_scipy():
    uniform = np.full((50, 10), 0.1)
    assert M.inception_score(uniform)[0] == pytest.approx(1.0) and M.inception_score(uniform)[1] == pytest.approx(0.0, abs=1e-9)
    one_hot = np.eye(10)[np.arange(200) % 10]
    assert M.inception_score(one_hot)[0] == pytest.approx(10.0, rel=1e-6)
    collapsed = np.eye(10)[np.zeros(200, dtype=int)]
    assert M.inception_score(collapsed)[0] == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(137, 10)) * 2
    probabilities = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    mean, std = M.inception_score(probabilities)
    expected_mean, expected_std = scipy_inception_score(probabilities)
    assert mean == pytest.approx(expected_mean, rel=1e-9) and std == pytest.approx(expected_std, rel=1e-9)
    for bad in (probabilities[:9], probabilities[0]):
        with pytest.raises(ValueError):
            M.inception_score(bad)


# Precision / recall

@pytest.mark.parametrize('k', [1, 3, 5])
@pytest.mark.parametrize('query_batch', [7, 1024])
def test_knn_radii_match_brute_force(k, query_batch):
    features = clouds()[1]
    distances = np.sort(brute_distances(features, features), axis=1)
    radii = M.knn_radii(features, k=k, query_batch=query_batch, device=CPU)
    np.testing.assert_allclose(radii, distances[:, k], rtol=1e-3)  # column 0 is the point itself
    with pytest.raises(ValueError):
        M.knn_radii(features[:3], k=3, device=CPU)


@pytest.mark.parametrize('query_batch', [11, 1024])
def test_precision_recall_match_brute_force(query_batch):
    generated, reference = clouds()
    distances = brute_distances(generated, reference)
    generated_radii = np.sort(brute_distances(generated, generated), axis=1)[:, 3]
    reference_radii = np.sort(brute_distances(reference, reference), axis=1)[:, 3]
    precision = (distances <= reference_radii[None, :]).any(axis=1).mean()
    recall = (distances <= generated_radii[:, None]).any(axis=0).mean()
    assert 0.05 < precision < 0.95 and 0.05 < recall < 0.95, 'the fixture must exercise both outcomes'
    got = M.precision_recall(generated, reference, M.knn_radii(generated, device=CPU), M.knn_radii(reference, device=CPU),
                             query_batch=query_batch, device=CPU)
    assert got == pytest.approx((precision, recall), abs=0.011)


def test_precision_recall_extremes_and_asymmetry():
    _, reference = clouds()
    radii = M.knn_radii(reference, device=CPU)
    assert M.precision_recall(reference, reference, radii, radii, device=CPU) == (1.0, 1.0)
    far = reference + 100.0
    assert M.precision_recall(far, reference, radii, radii, device=CPU) == (0.0, 0.0)
    # Mode collapse: realistic but concentrated samples keep precision and lose recall.
    collapsed = reference[:1] + 1e-3 * np.random.default_rng(1).normal(size=(60, reference.shape[1])).astype(np.float32)
    precision, recall = M.precision_recall(collapsed, reference, M.knn_radii(collapsed, device=CPU), radii, device=CPU)
    assert precision == 1.0 and recall < 0.05


# Assets, caching and the group breakdown

def make_metrics_root(root: Path, seed=0, dimensions=6):
    rng = np.random.default_rng(seed)
    labels = np.arange(600) % 100
    features = (rng.normal(size=(600, dimensions)) + labels[:, None] / 100).astype(np.float32)
    files = {'feature_file': ('feats.npy', features), 'labels_file': ('labels.npy', labels),
             'vgg_feature_file': ('vgg.npy', features), 'vgg_radii_file': ('radii.npy', np.ones(600, dtype=np.float32))}
    manifest = {}
    for key, (name, array) in files.items():
        np.save(root / name, array)
        manifest[key] = name
    np.savez(root / 'stats.npz', mu=features.astype(np.float64).mean(0), sigma=np.cov(features.astype(np.float64), rowvar=False))
    manifest['fid_file'] = 'stats.npz'
    for file_key, digest_key in M.ASSETS:
        manifest[digest_key] = sha256_file(root / manifest[file_key])
    (root / M.MANIFEST).write_text(json.dumps(manifest))
    return features, labels


def test_metric_asset_binding_verifies_every_file(tmp_path):
    make_metrics_root(tmp_path)
    binding = M.metric_asset_binding(tmp_path)
    assert len(binding['assets']) == 5 and binding['manifest_sha256'] == sha256_file(tmp_path / M.MANIFEST)
    assert binding['inception_source_sha256'] == sha256_file(M.INCEPTION_SOURCE)
    for name in ('feats.npy', 'labels.npy', 'stats.npz', 'vgg.npy', 'radii.npy'):
        original = (tmp_path / name).read_bytes()
        (tmp_path / name).write_bytes(original + b'\0')
        with pytest.raises(RuntimeError, match='hash mismatch'):
            M.metric_asset_binding(tmp_path)
        (tmp_path / name).write_bytes(original)
    assert M.metric_asset_binding(tmp_path) == binding


def test_fid_and_is_overall_and_per_group_match_scipy(tmp_path):
    reference, reference_labels = make_metrics_root(tmp_path)
    rng = np.random.default_rng(5)
    labels = np.arange(400) % 100
    features = (rng.normal(size=(400, 6)) * 1.1 + 0.2 + labels[:, None] / 50).astype(np.float32)
    logits = rng.normal(size=(400, 10)) + np.eye(10)[labels % 10] * 2
    probabilities = (np.exp(logits) / np.exp(logits).sum(1, keepdims=True)).astype(np.float32)
    result = M.fid_and_is(features, probabilities, labels, tmp_path, CPU)

    moments = lambda values: (values.astype(np.float64).mean(0), np.cov(values.astype(np.float64), rowvar=False))
    assert result['FID']['Overall'] == pytest.approx(scipy_fid(*moments(features), *moments(reference)), rel=1e-6)
    assert result['IS']['Overall']['mean'] == pytest.approx(scipy_inception_score(probabilities.astype(np.float64))[0], rel=1e-6)
    for name, classes in (('Many', range(0, 33)), ('Medium', range(33, 66)), ('Few', range(66, 100))):
        mine, theirs = np.isin(labels, list(classes)), np.isin(reference_labels, list(classes))
        assert result['FID'][name] == pytest.approx(scipy_fid(*moments(features[mine]), *moments(reference[theirs])), rel=1e-6)
        expected = scipy_inception_score(probabilities[mine].astype(np.float64))
        assert result['IS'][name]['mean'] == pytest.approx(expected[0], rel=1e-6)
        assert result['IS'][name]['std'] == pytest.approx(expected[1], rel=1e-5, abs=1e-9)
    assert set(result) == {'FID', 'IS'} and set(result['FID']) == {'Overall', 'Many', 'Medium', 'Few'}


def test_fid_ranks_a_matching_sample_below_a_shifted_one(tmp_path):
    """Sensitivity of the instrument on synthetic features."""
    reference, labels = make_metrics_root(tmp_path)
    rng = np.random.default_rng(6)
    probabilities = np.full((600, 10), 0.1, dtype=np.float32)
    matching = (rng.normal(size=(600, 6)) + labels[:, None] / 100).astype(np.float32)
    fid = lambda values: M.fid_and_is(values, probabilities, labels, tmp_path, CPU)['FID']
    assert all(fid(matching)[name] < 0.25 * fid(matching + 0.5)[name] for name in ('Overall', 'Many', 'Medium', 'Few'))
    assert fid(reference)['Overall'] == pytest.approx(0.0, abs=1e-6)


def test_inception_cache_is_keyed_by_the_samples_hash(tmp_path, monkeypatch):
    calls = []

    def fake_statistics(images, *, batch_size, device):
        calls.append(len(images))
        flat = np.asarray(images).reshape(len(images), -1)
        return flat[:, :4].astype(np.float32), np.full((len(images), 3), 1 / 3, dtype=np.float32)

    monkeypatch.setattr(M, 'inception_statistics', fake_statistics)
    samples = tmp_path / 'samples.npy'
    np.save(samples, np.random.default_rng(0).random((12, 4, 4, 3), dtype=np.float32))
    first = M.cached_inception_statistics(samples, batch_size=4, device=CPU)
    second = M.cached_inception_statistics(samples, batch_size=4, device=CPU)
    assert calls == [12] and (tmp_path / 'samples.inception.npz').is_file()
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.save(samples, np.random.default_rng(1).random((12, 4, 4, 3), dtype=np.float32))
    third = M.cached_inception_statistics(samples, batch_size=4, device=CPU)
    assert calls == [12, 12] and not np.array_equal(first[0], third[0]), 'a stale cache must not be reused'


def test_compute_metrics_without_recall_uses_only_inception(tmp_path, monkeypatch):
    make_metrics_root(tmp_path)
    labels = np.arange(400) % 100
    rng = np.random.default_rng(2)
    samples = tmp_path / 'samples.npy'
    np.save(samples, rng.random((400, 2, 2, 3), dtype=np.float32))
    fake = lambda images, *, batch_size, device: (np.asarray(images).reshape(len(images), -1)[:, :6].astype(np.float32),
                                                  np.full((len(images), 10), 0.1, dtype=np.float32))
    monkeypatch.setattr(M, 'inception_statistics', fake)
    monkeypatch.setattr(M, 'vgg16_fc2', lambda *a, **k: pytest.fail('VGG must not run when recall is off'))
    result = M.compute_metrics(samples, labels, {'inception_batch_size': 8, 'vgg_batch_size': 8, 'recall': False}, tmp_path, CPU)
    assert set(result) == {'FID', 'IS'} and np.isfinite(result['FID']['Overall'])

    monkeypatch.setattr(M, 'vgg16_fc2', lambda images, *, batch_size, device: fake(images, batch_size=0, device=None)[0])
    full = M.compute_metrics(samples, labels, {'inception_batch_size': 8, 'vgg_batch_size': 8, 'recall': True}, tmp_path, CPU)
    assert set(full) == {'FID', 'IS', 'Precision', 'Recall'} and 0.0 <= full['Precision'] <= 1.0 and 0.0 <= full['Recall'] <= 1.0


def test_inception_statistics_rejects_non_nhwc_input():
    with pytest.raises(ValueError):
        M.inception_statistics(np.zeros((2, 3, 32, 32), dtype=np.float32), batch_size=2, device=CPU)


# Positive controls on the pinned assets

def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def need(*paths):
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        pytest.skip(f'missing assets: {missing}')


@pytest.fixture(scope='module')
def manifest():
    need(STATS / M.MANIFEST)
    return json.loads((STATS / M.MANIFEST).read_text())


@pytest.mark.real_assets
def test_pinned_assets_verify_against_their_manifest(manifest):
    binding = M.metric_asset_binding(STATS)
    assert len(binding['assets']) == 5


@pytest.mark.real_assets
def test_stored_fid_statistics_are_the_moments_of_the_stored_features(manifest):
    """If mu/sigma and the per-class feature bank disagreed, overall and group FID would not be comparable."""
    features = np.load(STATS / manifest['feature_file'], mmap_mode='r')
    with np.load(STATS / manifest['fid_file']) as statistics:
        stored = (torch.from_numpy(statistics['mu']), torch.from_numpy(statistics['sigma']))
    distance = M.frechet_distance(*M._moments(np.asarray(features), CPU), *stored)
    print(f'FID(stored features, stored mu/sigma) = {distance:.3e}')
    assert abs(distance) < 1e-2


@pytest.mark.real_assets
def test_inception_extractor_reproduces_the_stored_reference_features(manifest):
    need(DATA, HUB / 'pt_inception-2015-12-05-6726825d.pth')
    import torchvision
    train = torchvision.datasets.CIFAR100(str(REPOSITORY / 'data'), train=True, download=False)
    stored_labels = np.load(STATS / manifest['labels_file'])
    assert (stored_labels == np.asarray(train.targets)).all(), 'the reference bank is in dataset order'
    images = train.data[:256].astype(np.float32) / 255.0
    features, probabilities = M.inception_statistics(images, batch_size=64, device=device())
    stored = np.load(STATS / manifest['feature_file'], mmap_mode='r')[:256]
    error = np.abs(features - stored).max()
    print(f'max |feature - stored| = {error:.3e} (feature scale {np.abs(stored).mean():.3f})')
    assert features.shape == (256, 2048) and error < 2e-2
    assert probabilities.shape[0] == 256 and np.allclose(probabilities.sum(1), 1.0, atol=1e-3)
    with pytest.raises(ValueError):
        M.inception_statistics(images * 255.0, batch_size=64, device=device())


@pytest.mark.real_assets
def test_fid_and_is_separate_real_images_from_corrupted_ones(manifest):
    """Instrument sensitivity: held-out real CIFAR-100 must score far better than a noised copy."""
    need(DATA, HUB / 'pt_inception-2015-12-05-6726825d.pth')
    import torchvision
    test = torchvision.datasets.CIFAR100(str(REPOSITORY / 'data'), train=False, download=False)
    targets = np.asarray(test.targets)
    chosen = np.concatenate([np.flatnonzero(targets == c)[:50] for c in range(100)])
    images, labels = test.data[chosen].astype(np.float32) / 255.0, targets[chosen]
    noisy = np.clip(images + np.random.default_rng(0).normal(scale=0.1, size=images.shape), 0, 1).astype(np.float32)
    score = lambda values: M.fid_and_is(*M.inception_statistics(values, batch_size=100, device=device()), labels, STATS, device())
    real, corrupted = score(images), score(noisy)
    print('real test images :', json.dumps(real))
    print('noised (sd 0.1)  :', json.dumps(corrupted))
    assert real['FID']['Overall'] < 12.0
    assert all(corrupted['FID'][name] > 3 * real['FID'][name] for name in real['FID'])
    assert real['IS']['Overall']['mean'] > corrupted['IS']['Overall']['mean'] > 1.0


@pytest.mark.real_assets
def test_stored_vgg_radii_are_the_k3_radii_of_the_stored_vgg_features(manifest):
    features = np.load(STATS / manifest['vgg_feature_file'], mmap_mode='r')
    stored = np.load(STATS / manifest['vgg_radii_file'])
    try:
        radii = M.knn_radii(np.asarray(features), device=device())
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        radii = M.knn_radii(np.asarray(features), device=CPU)
    relative = np.abs(radii - stored) / stored
    print(f'k=3 radii: max relative difference {relative.max():.3e}, median {np.median(relative):.3e}')
    assert np.median(relative) < 1e-3 and np.quantile(relative, 0.999) < 2e-2


@pytest.mark.real_assets
def test_precision_recall_controls_on_the_stored_vgg_features(manifest):
    """Sensitivity of the P/R instrument, measured on real features (second half = bank, first half = 'generated').

    Measured on this machine: balanced (0.79, 0.80); 10 of 100 classes (0.78, 0.74); a single class
    (0.79, 0.62); variance shrunk to 30 % (1.00, 0.00); features plus noise (0.00, 1.00).
    Recall reacts strongly to a loss of spread but only weakly to missing classes, so it is not a
    class-coverage measure for the Many / Medium / Few question.
    """
    features = np.load(STATS / manifest['vgg_feature_file'], mmap_mode='r')
    labels = np.load(STATS / manifest['labels_file'])[:25000]
    bank, first = np.asarray(features[25000:]), np.asarray(features[:25000])
    bank_radii = M.knn_radii(bank, device=device())
    rng = np.random.default_rng(0)
    score = lambda generated: M.precision_recall(generated, bank, M.knn_radii(generated, device=device()), bank_radii,
                                                 device=device())
    balanced = first[rng.permutation(25000)[:2500]]
    ten = first[labels < 10]
    one = first[labels < 1]
    centre = balanced.mean(0)
    results = {'balanced': score(balanced), 'ten classes': score(ten), 'one class': score(one),
               'shrunk': score((centre + 0.3 * (balanced - centre)).astype(np.float32)),
               'noised': score((balanced + rng.normal(size=balanced.shape) * balanced.std()).astype(np.float32))}
    print('precision/recall controls:', {name: tuple(round(v, 4) for v in value) for name, value in results.items()})
    assert results['balanced'][1] > results['ten classes'][1] > results['one class'][1]
    assert results['shrunk'][0] > 0.95 and results['shrunk'][1] < 0.05
    assert results['noised'][0] < 0.05
    assert 0.6 < results['balanced'][0] < 0.95 and 0.6 < results['balanced'][1] < 0.95
