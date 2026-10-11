"""Closed-form checks of the metric formulas (no pretrained feature extractors needed)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scaffold.metrics import _moments, frechet_distance, inception_score, knn_radii, precision_recall

DEVICE = torch.device('cpu')


def test_frechet_distance_closed_forms():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(500, 6))
    mean, covariance = _moments(features, DEVICE)
    assert abs(frechet_distance(mean, covariance, mean, covariance)) < 1e-8
    shift = torch.full_like(mean, 0.5)
    assert frechet_distance(mean, covariance, mean + shift, covariance) == pytest.approx(6 * 0.25, abs=1e-8)
    # Isotropic case: d * (sigma_a - sigma_b)^2.
    eye = torch.eye(6, dtype=torch.float64)
    assert frechet_distance(mean, 4 * eye, mean, eye) == pytest.approx(6.0, abs=1e-9)
    assert np.allclose(covariance.numpy(), np.cov(features, rowvar=False))


def test_inception_score_bounds():
    uniform = np.full((100, 10), 0.1)
    assert inception_score(uniform)[0] == pytest.approx(1.0)
    one_hot = np.eye(10)[np.arange(100) % 10]
    assert inception_score(one_hot)[0] == pytest.approx(10.0, rel=1e-6)
    with pytest.raises(ValueError):
        inception_score(uniform[:5])


def test_knn_radii_and_precision_recall():
    line = np.arange(6, dtype=np.float32).reshape(6, 1) * np.array([[1.0, 0.0]], dtype=np.float32)
    radii = knn_radii(line, k=2, device=DEVICE)
    assert np.allclose(radii, [2, 1, 1, 1, 1, 2])
    precision, recall = precision_recall(line, line, radii, radii, device=DEVICE)
    assert precision == 1.0 and recall == 1.0
    far = line + np.array([[0.0, 100.0]], dtype=np.float32)
    assert precision_recall(far, line, radii, radii, device=DEVICE) == (0.0, 0.0)
    with pytest.raises(ValueError):
        knn_radii(line[:2], k=2, device=DEVICE)
