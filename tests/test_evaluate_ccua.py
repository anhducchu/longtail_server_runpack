import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = ROOT / "tools" / "evaluate_ccua.py"


def _evaluator_module():
    spec = importlib.util.spec_from_file_location("evaluate_ccua_under_test", EVALUATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scipy_fid(features: np.ndarray, reference: np.ndarray) -> float:
    scipy_linalg = pytest.importorskip("scipy.linalg")
    generated_mean = np.mean(features, axis=0, dtype=np.float64)
    reference_mean = np.mean(reference, axis=0, dtype=np.float64)
    generated_covariance = np.cov(features, rowvar=False)
    reference_covariance = np.cov(reference, rowvar=False)
    covariance_mean = scipy_linalg.sqrtm(generated_covariance.dot(reference_covariance))
    if np.iscomplexobj(covariance_mean):
        assert np.allclose(np.diagonal(covariance_mean).imag, 0, atol=1e-3)
        covariance_mean = covariance_mean.real
    delta = generated_mean - reference_mean
    return float(
        delta.dot(delta)
        + np.trace(generated_covariance)
        + np.trace(reference_covariance)
        - 2 * np.trace(covariance_mean)
    )


@pytest.mark.parametrize(
    "generated_count,reference_count,dimension",
    [(3, 4, 8), (6, 5, 12), (9, 7, 16)],
)
def test_low_rank_fid_matches_scipy_sqrtm(generated_count, reference_count, dimension):
    evaluator = _evaluator_module()
    rng = np.random.default_rng(1000 + generated_count + reference_count + dimension)
    generated = rng.normal(size=(generated_count, dimension))
    reference = rng.normal(size=(reference_count, dimension))

    assert evaluator._should_use_low_rank_fid(generated, reference)
    assert evaluator._fid_low_rank(generated, reference) == pytest.approx(
        _scipy_fid(generated, reference), rel=2e-7, abs=2e-6
    )


def _install_fake_fid_module(monkeypatch, value: float) -> None:
    score = types.ModuleType("score")
    score.__path__ = []
    fid = types.ModuleType("score.fid")
    fid.calculate_frechet_distance = lambda *_: value
    monkeypatch.setitem(sys.modules, "score", score)
    monkeypatch.setitem(sys.modules, "score.fid", fid)


def _args(tmp_path: Path, *, mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        repo=tmp_path / "repo",
        metrics_root=tmp_path / "metrics",
        output=tmp_path / "metrics.json",
        samples=None,
        generated_features_cache=None,
        metric_backend="canonical",
        inception_batch_size=16,
        vgg_batch_size=128,
        knn_query_batch=1024,
        mode=mode,
        per_class_output=None,
        device="cpu",
    )


def test_headline_metrics_reuse_features_and_include_seeded_kid(tmp_path, monkeypatch):
    evaluator = _evaluator_module()
    args = _args(tmp_path, mode="detailed")
    args.metrics_root.mkdir()
    generated = np.arange(40, dtype=np.float32).reshape(10, 4)
    reference = generated + 0.5
    np.save(args.metrics_root / "cifar100_feats.npy", reference)
    np.savez(args.metrics_root / "cifar100.train.npz", mu=np.zeros(4), sigma=np.eye(4))
    _install_fake_fid_module(monkeypatch, 12.5)
    monkeypatch.setattr(
        evaluator,
        "_cached_inception_features_and_is",
        lambda *_: (generated, (8.5, 0.2)),
    )
    monkeypatch.setattr(evaluator, "_standard_prd_scores", lambda *_: (0.61, 0.42))
    calls = []

    def fake_kid(generated_features, reference_features, *, num_subsets, max_subset_size, rng):
        calls.append((generated_features, reference_features, num_subsets, max_subset_size, rng.integers(1_000_000)))
        return 0.0125

    monkeypatch.setattr(evaluator, "polynomial_mmd_kid", fake_kid)
    metrics, returned_generated, returned_reference = evaluator._collect_headline_metrics(
        args, images=object()
    )

    assert returned_generated is generated
    np.testing.assert_allclose(returned_reference, reference)
    assert set(metrics) == {"FID", "IS", "IS_std", "F_8", "F_1_8", "KID"}
    assert calls[0][2:4] == (100, 1000)
    assert calls[0][4] == np.random.default_rng(2026).integers(1_000_000)


@pytest.mark.parametrize("mode", ("detailed", "gpu_fast", "fixed_table"))
def test_evaluator_rejects_non_50k_samples(tmp_path, monkeypatch, mode):
    evaluator = _evaluator_module()
    samples = tmp_path / "samples.npy"
    labels = tmp_path / "labels.npy"
    np.save(samples, np.zeros((20, 3, 32, 32), dtype=np.uint8))
    np.save(labels, np.arange(20, dtype=np.int64))
    monkeypatch.setattr(evaluator, "_evaluate", lambda *_: pytest.fail("must fail first"))
    monkeypatch.setattr(sys, "argv", [
        "evaluate_ccua.py",
        "--repo", str(tmp_path / "repo"),
        "--samples", str(samples),
        "--labels", str(labels),
        "--metrics-root", str(tmp_path / "metrics"),
        "--output", str(tmp_path / "metrics.json"),
        "--mode", mode,
    ])

    with pytest.raises(ValueError, match="CIFAR-100 protocol requires 50k"):
        evaluator.main()


def test_detailed_publishes_headline_before_vgg_and_passes_device(tmp_path, monkeypatch):
    evaluator = _evaluator_module()
    args = _args(tmp_path, mode="detailed")
    args.metrics_root.mkdir()
    headline_metrics = {"FID": 1.0, "IS": 2.0, "IS_std": 0.1, "F_8": 0.3, "F_1_8": 0.4}
    monkeypatch.setattr(
        evaluator,
        "_collect_headline_metrics",
        lambda *_: (headline_metrics, np.zeros((2, 4)), np.zeros((2, 4))),
    )
    np.save(args.metrics_root / "cifar100_vgg16_fc2.npy", np.zeros((2, 4)))
    np.save(args.metrics_root / "cifar100_vgg16_fc2_k3_radii.npy", np.zeros(2))

    def fake_improved_prd(*_args, **kwargs):
        early = json.loads(args.output.read_text(encoding="utf-8"))
        assert early["protocol"]["evaluation_mode"] == "headline"
        assert "ImprovedPrecision" not in early["metrics"]
        assert kwargs["device"] == "cpu"
        return 0.7, 0.8

    monkeypatch.setattr(evaluator, "improved_prd_vgg16_k3", fake_improved_prd)
    payload = evaluator._evaluate(
        args,
        images=np.zeros((2, 3, 32, 32)),
        labels=np.zeros(2, dtype=np.int64),
        counts=np.zeros(100, dtype=np.int64),
    )

    assert payload["metrics"]["ImprovedPrecision"] == 0.7
    assert payload["metrics"]["Recall"] == 0.8
    assert "evaluation_mode" not in payload["protocol"]


def test_generated_inception_cache_is_reused_for_same_sample(tmp_path, monkeypatch):
    evaluator = _evaluator_module()
    samples = tmp_path / "samples.npy"
    np.save(samples, np.zeros((3, 3, 32, 32), dtype=np.uint8))
    args = _args(tmp_path, mode="detailed")
    args.samples = samples
    args.generated_features_cache = tmp_path / "inception_features.npz"
    features = np.zeros((3, 2048), dtype=np.float32)
    calls = []

    def fake_extract(repo, images, batch_size, *, device):
        calls.append((repo, images, batch_size, device))
        return features, (4.0, 0.2)

    monkeypatch.setattr(evaluator, "_inception_features_and_is", fake_extract)
    first = evaluator._cached_inception_features_and_is(args, np.zeros((3, 3, 32, 32)))
    second = evaluator._cached_inception_features_and_is(args, np.zeros((3, 3, 32, 32)))

    assert len(calls) == 1
    np.testing.assert_array_equal(first[0], features)
    np.testing.assert_array_equal(second[0], features)
    assert second[1] == (4.0, 0.2)


def test_gpu_fast_publishes_separate_preview(tmp_path, monkeypatch):
    evaluator = _evaluator_module()
    args = _args(tmp_path, mode="gpu_fast")
    args.per_class_output = tmp_path / "fast_per_class.json"
    fast_metrics = {"FID": 8.0, "IS": 7.0, "IS_std": 0.1, "KID": 0.02}
    monkeypatch.setattr(
        evaluator,
        "_collect_gpu_fast_metrics",
        lambda *_: (dict(fast_metrics), np.zeros((20, 4)), np.zeros((40, 4))),
    )
    monkeypatch.setattr(
        evaluator,
        "_per_class_payload",
        lambda *_: {"groups": {"Many": {"FID": 1.0}, "Medium": {"FID": 2.0}, "Few": {"FID": 3.0}}},
    )

    payload = evaluator._evaluate(
        args,
        images=np.zeros((20, 3, 32, 32)),
        labels=np.zeros(20, dtype=np.int64),
        counts=np.zeros(100, dtype=np.int64),
    )

    assert payload["protocol"]["evaluation_mode"] == "gpu_fast"
    assert payload["protocol"]["paper_report_ready"] is False
    assert payload["metrics"]["Few_FID"] == 3.0
    assert json.loads(args.output.read_text(encoding="utf-8")) == payload
    assert args.per_class_output.exists()


def test_fixed_table_computes_only_metrics_used_by_the_table(tmp_path, monkeypatch):
    evaluator = _evaluator_module()
    args = _args(tmp_path, mode="fixed_table")
    args.per_class_output = tmp_path / "fixed_per_class.json"
    args.metrics_root.mkdir()
    np.save(args.metrics_root / "cifar100_vgg16_fc2.npy", np.zeros((2, 4)))
    np.save(args.metrics_root / "cifar100_vgg16_fc2_k3_radii.npy", np.zeros(2))
    monkeypatch.setattr(
        evaluator,
        "_collect_fixed_table_metrics",
        lambda *_: ({"FID": 9.0}, np.zeros((2, 4)), np.zeros((2, 4))),
    )
    monkeypatch.setattr(
        evaluator,
        "improved_prd_vgg16_k3",
        lambda *_args, **_kwargs: (0.7, 0.8),
    )
    monkeypatch.setattr(
        evaluator,
        "_per_class_payload",
        lambda *_: {
            "groups": {
                "Many": {"FID": 1.0},
                "Medium": {"FID": 2.0},
                "Few": {"FID": 3.0},
            }
        },
    )

    payload = evaluator._evaluate(
        args,
        images=np.zeros((2, 3, 32, 32)),
        labels=np.zeros(2, dtype=np.int64),
        counts=np.zeros(100, dtype=np.int64),
    )

    assert payload["metrics"] == {
        "FID": 9.0,
        "ImprovedPrecision": 0.7,
        "Recall": 0.8,
    }
    assert payload["protocol"]["evaluation_mode"] == "fixed_table"
    assert set(payload["protocol"]["omitted_metrics"]) == {
        "IS",
        "IS_std",
        "F_8",
        "F_1_8",
        "KID",
    }
    assert args.per_class_output.exists()
