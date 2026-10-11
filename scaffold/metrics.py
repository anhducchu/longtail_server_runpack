"""Image-quality metrics against pinned CIFAR-100 reference assets.

    FID, IS            pinned CCUA-DDPM InceptionV3 (2048-d pool features, class probabilities)
    Precision, Recall  improved PRD (Kynkaanniemi et al.) on torchvision VGG-16 fc2, k = 3

Reference statistics live in ``evaluation.metrics_root`` and are listed, with
their SHA-256, in ``cifar100.metric_manifest.json``. Every asset is verified
before any sample is generated. Many / Medium / Few are the class-index thirds
of the long-tailed split, sorted from the most to the least frequent class.
"""
from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scaffold.utils import atomic_npz, sha256_file

GROUPS = {'Many': np.arange(0, 33), 'Medium': np.arange(33, 66), 'Few': np.arange(66, 100)}
MANIFEST = 'cifar100.metric_manifest.json'
ASSETS = (('fid_file', 'fid_sha256'), ('feature_file', 'feature_sha256'), ('labels_file', 'labels_sha256'),
          ('vgg_feature_file', 'vgg_feature_sha256'), ('vgg_radii_file', 'vgg_radii_sha256'))
INCEPTION_SOURCE = Path(__file__).resolve().parents[1] / 'third_party' / 'CCUA-DDPM' / 'score' / 'inception.py'


def metric_asset_binding(metrics_root: Path) -> dict[str, Any]:
    """Verify every reference asset against the manifest and return their fingerprint."""
    manifest = json.loads((metrics_root / MANIFEST).read_text(encoding='utf-8'))
    for file_key, digest_key in ASSETS:
        if sha256_file(metrics_root / manifest[file_key]) != manifest[digest_key]:
            raise RuntimeError(f'metric asset hash mismatch: {manifest[file_key]}')
    return {'manifest_sha256': sha256_file(metrics_root / MANIFEST),
            'assets': {manifest[file_key]: manifest[digest_key] for file_key, digest_key in ASSETS},
            'inception_source_sha256': sha256_file(INCEPTION_SOURCE)}


# Inception: FID and IS

@torch.no_grad()
def inception_statistics(images: np.ndarray, *, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Pool features [N, 2048] and class probabilities [N, K] for NHWC images in [0, 1]."""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f'expected NHWC RGB images, got {images.shape}')
    inception = import_module('third_party.CCUA-DDPM.score.inception').InceptionV3
    blocks = [inception.BLOCK_INDEX_BY_DIM[2048], inception.BLOCK_INDEX_BY_DIM['prob']]
    model = inception(blocks).to(device).eval()
    features, probabilities = [], []
    for start in range(0, len(images), batch_size):
        batch = np.ascontiguousarray(images[start:start + batch_size].transpose(0, 3, 1, 2))
        if batch.min() < 0.0 or batch.max() > 1.0:
            raise ValueError('Inception inputs must lie in [0, 1]')
        feature, probability = model(torch.from_numpy(batch).float().to(device))
        features.append(feature.flatten(1).cpu().numpy().astype(np.float32))
        probabilities.append(probability.cpu().numpy().astype(np.float32))
    return np.concatenate(features), np.concatenate(probabilities)


def cached_inception_statistics(samples: Path, *, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Inception statistics of a samples file, cached next to it and keyed by its SHA-256."""
    cache = samples.with_name(samples.stem + '.inception.npz')
    digest = sha256_file(samples)
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as loaded:
            if str(loaded['samples_sha256']) == digest:
                return loaded['features'], loaded['probabilities']
    features, probabilities = inception_statistics(np.load(samples, mmap_mode='r'), batch_size=batch_size, device=device)
    atomic_npz(cache, features=features, probabilities=probabilities, samples_sha256=np.asarray(digest))
    return features, probabilities


def inception_score(probabilities: np.ndarray, *, splits: int = 10) -> tuple[float, float]:
    """Mean and standard deviation of exp(E KL(p(y|x) || p(y))) over contiguous splits."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(probabilities) < splits:
        raise ValueError(f'IS needs a [N, K] probability array with N >= {splits}')
    scores = []
    for split in range(splits):
        block = probabilities[split * len(probabilities) // splits:(split + 1) * len(probabilities) // splits]
        marginal = np.clip(block.mean(axis=0), 1e-12, 1.0)
        block = np.clip(block, 1e-12, 1.0)
        scores.append(np.exp(np.mean(np.sum(block * (np.log(block) - np.log(marginal)), axis=1))))
    return float(np.mean(scores)), float(np.std(scores))


def _moments(features: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and unbiased covariance in float64."""
    if features.ndim != 2 or len(features) < 2:
        raise ValueError(f'FID needs at least two feature vectors, got {features.shape}')
    values = torch.as_tensor(np.ascontiguousarray(features, dtype=np.float64), device=device)
    mean = values.mean(dim=0)
    centered = values - mean
    return mean, centered.T @ centered / (len(values) - 1)


def frechet_distance(mean_a: torch.Tensor, covariance_a: torch.Tensor,
                     mean_b: torch.Tensor, covariance_b: torch.Tensor) -> float:
    """|mu_a - mu_b|^2 + tr(C_a) + tr(C_b) - 2 tr((C_a^1/2 C_b C_a^1/2)^1/2), via symmetric eigensolves."""
    covariance_a = (covariance_a + covariance_a.T) / 2
    covariance_b = (covariance_b + covariance_b.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_a)
    root_a = eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(0) @ eigenvectors.T
    middle = root_a @ covariance_b @ root_a
    cross = torch.linalg.eigvalsh((middle + middle.T) / 2).clamp_min(0).sqrt().sum()
    distance = (mean_a - mean_b).square().sum() + torch.trace(covariance_a) + torch.trace(covariance_b) - 2 * cross
    return float(distance)


def fid_and_is(features: np.ndarray, probabilities: np.ndarray, labels: np.ndarray,
               metrics_root: Path, device: torch.device) -> dict[str, Any]:
    """FID and IS overall and per frequency group.

    Overall FID uses the stored full-reference statistics; group FID compares
    the group's generated features with the reference features of its classes.
    """
    manifest = json.loads((metrics_root / MANIFEST).read_text(encoding='utf-8'))
    reference = np.load(metrics_root / manifest['feature_file'], mmap_mode='r')
    reference_labels = np.load(metrics_root / manifest['labels_file'])
    with np.load(metrics_root / manifest['fid_file']) as statistics:
        reference_moments = (torch.as_tensor(statistics['mu'], dtype=torch.float64, device=device),
                             torch.as_tensor(statistics['sigma'], dtype=torch.float64, device=device))
    mean, std = inception_score(probabilities)
    fid = {'Overall': frechet_distance(*_moments(features, device), *reference_moments)}
    score = {'Overall': {'mean': mean, 'std': std}}
    for name, classes in GROUPS.items():
        members = np.isin(labels, classes)
        fid[name] = frechet_distance(*_moments(features[members], device),
                                     *_moments(reference[np.isin(reference_labels, classes)], device))
        mean, std = inception_score(probabilities[members])
        score[name] = {'mean': mean, 'std': std}
    return {'FID': fid, 'IS': score}


# VGG-16: improved precision and recall

@torch.no_grad()
def vgg16_fc2(images: np.ndarray, *, batch_size: int, device: torch.device) -> np.ndarray:
    """fc2 activations [N, 4096] of ImageNet VGG-16 for NHWC images in [0, 1], resized to 224.

    No ImageNet mean/std normalisation is applied, matching the extractor that
    produced the stored reference features.
    """
    from torchvision import models
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device).eval()
    features = np.empty((len(images), 4096), dtype=np.float32)
    for start in range(0, len(images), batch_size):
        batch = np.ascontiguousarray(images[start:start + batch_size].transpose(0, 3, 1, 2))
        batch = torch.nn.functional.interpolate(torch.from_numpy(batch).float().to(device), size=(224, 224))
        features[start:start + len(batch)] = model.classifier[:4](model.features(batch).flatten(1)).cpu().numpy()
    return features


def _squared_distances(query: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    return (query.square().sum(dim=1, keepdim=True) + bank.square().sum(dim=1) - 2 * query @ bank.T).clamp_min_(0)


@torch.no_grad()
def knn_radii(features: np.ndarray, *, k: int = 3, query_batch: int = 1024, device: torch.device) -> np.ndarray:
    """Distance from each feature to its k-th nearest neighbour, excluding itself."""
    if len(features) <= k:
        raise ValueError(f'k-NN radii need more than k samples, got n={len(features)}, k={k}')
    bank = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(device)
    radii = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(bank), query_batch):
        distances = _squared_distances(bank[start:start + query_batch], bank)
        rows = torch.arange(len(distances), device=device)
        distances[rows, start + rows] = float('inf')
        radii[start:start + len(distances)] = torch.topk(distances, k=k, largest=False).values[:, -1].sqrt().cpu().numpy()
    return radii


@torch.no_grad()
def precision_recall(generated: np.ndarray, reference: np.ndarray, generated_radii: np.ndarray,
                     reference_radii: np.ndarray, *, query_batch: int = 1024, device: torch.device) -> tuple[float, float]:
    """Precision: generated points inside some reference ball. Recall: the converse."""
    bank = torch.from_numpy(np.ascontiguousarray(reference, dtype=np.float32)).to(device)
    bank_radii = torch.from_numpy(np.ascontiguousarray(reference_radii, dtype=np.float32)).to(device).square()
    queries = torch.from_numpy(np.ascontiguousarray(generated, dtype=np.float32)).to(device)
    query_radii = torch.from_numpy(np.ascontiguousarray(generated_radii, dtype=np.float32)).to(device).square()
    precise = torch.zeros(len(queries), dtype=torch.bool, device=device)
    recalled = torch.zeros(len(bank), dtype=torch.bool, device=device)
    for start in range(0, len(queries), query_batch):
        chunk = slice(start, start + query_batch)
        distances = _squared_distances(queries[chunk], bank)
        precise[chunk] = (distances <= bank_radii.unsqueeze(0)).any(dim=1)
        recalled |= (distances <= query_radii[chunk].unsqueeze(1)).any(dim=0)
    return float(precise.float().mean()), float(recalled.float().mean())


# Optional class accuracy

@torch.no_grad()
def classifier_accuracy(images: np.ndarray, labels: np.ndarray, settings: dict[str, Any],
                        device: torch.device) -> dict[str, float]:
    """Top-1 accuracy of an external CIFAR-100 classifier on the generated labels, overall and per group.

    ``settings`` names a TorchScript module (``torchscript``, ``sha256``) that maps
    NCHW images normalised with ``mean`` / ``std`` to 100 logits. The classifier
    is not part of this repository and must be validated separately; the number
    measures agreement with that classifier, not class coverage.
    """
    path = Path(settings['torchscript'])
    if sha256_file(path) != settings['sha256']:
        raise RuntimeError(f'classifier SHA mismatch: {path}')
    model = torch.jit.load(str(path), map_location=device).eval()
    mean = torch.tensor(settings['mean'], device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(settings['std'], device=device).reshape(1, 3, 1, 1)
    batch_size = int(settings.get('batch_size', 256))
    predictions = []
    for start in range(0, len(images), batch_size):
        batch = np.ascontiguousarray(images[start:start + batch_size].transpose(0, 3, 1, 2))
        logits = model((torch.from_numpy(batch).float().to(device) - mean) / std)
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    correct = np.concatenate(predictions) == labels
    result = {'Overall': float(correct.mean())}
    for name, classes in GROUPS.items():
        result[name] = float(correct[np.isin(labels, classes)].mean())
    return result


# Entry point used by evaluation

def compute_metrics(samples: Path, labels: np.ndarray, settings: dict[str, Any],
                    metrics_root: Path, device: torch.device) -> dict[str, Any]:
    """All metrics for one samples file (float32 NHWC in [0, 1]) with its class labels."""
    features, probabilities = cached_inception_statistics(
        samples, batch_size=int(settings['inception_batch_size']), device=device)
    result = fid_and_is(features, probabilities, labels, metrics_root, device)
    if settings['recall']:
        manifest = json.loads((metrics_root / MANIFEST).read_text(encoding='utf-8'))
        generated = vgg16_fc2(np.load(samples, mmap_mode='r'), batch_size=int(settings['vgg_batch_size']), device=device)
        precision, recall = precision_recall(
            generated, np.load(metrics_root / manifest['vgg_feature_file'], mmap_mode='r'),
            knn_radii(generated, device=device), np.load(metrics_root / manifest['vgg_radii_file']), device=device)
        result.update(Precision=precision, Recall=recall)
    if settings.get('classifier'):
        result['ClassAccuracy'] = classifier_accuracy(np.load(samples, mmap_mode='r'), labels,
                                                      settings['classifier'], device)
    return result
