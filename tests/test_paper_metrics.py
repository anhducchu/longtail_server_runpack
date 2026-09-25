import numpy as np
import pytest
import torch

from runner import metrics as paper_metrics


def _numpy_radii(features: np.ndarray, k: int) -> np.ndarray:
    distances = ((features[:, None, :] - features[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(distances, np.inf)
    return np.sqrt(np.sort(distances, axis=1)[:, k - 1])


def _numpy_membership(query: np.ndarray, manifold: np.ndarray, radii: np.ndarray) -> float:
    distances = ((query[:, None, :] - manifold[None, :, :]) ** 2).sum(axis=2)
    return float(np.mean((distances <= radii[None, :] ** 2).any(axis=1)))


def test_torch_knn_radii_matches_exact_numpy_reference_on_cpu():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(9, 5)).astype(np.float32)

    actual = paper_metrics._knn_radii_torch(
        features, k=3, query_batch=2, device=torch.device("cpu")
    )

    np.testing.assert_allclose(actual, _numpy_radii(features, k=3), rtol=1e-6, atol=1e-6)


def test_fused_cross_membership_matches_two_exact_directions_on_cpu():
    rng = np.random.default_rng(11)
    generated = rng.normal(size=(7, 4)).astype(np.float32)
    reference = rng.normal(size=(8, 4)).astype(np.float32)
    generated_radii = _numpy_radii(generated, k=3)
    reference_radii = _numpy_radii(reference, k=3)

    actual_precision, actual_recall = paper_metrics._fused_cross_membership_torch(
        generated,
        reference,
        generated_radii,
        reference_radii,
        query_batch=2,
        device=torch.device("cpu"),
    )

    expected_precision = _numpy_membership(generated, reference, reference_radii)
    expected_recall = _numpy_membership(reference, generated, generated_radii)
    assert actual_precision == pytest.approx(expected_precision, abs=1e-7)
    assert actual_recall == pytest.approx(expected_recall, abs=1e-7)
