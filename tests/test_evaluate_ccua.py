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
    """The scipy.sqrtm branch from the pinned score.fid implementation."""
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


@pytest.mark.parametrize("generated_count,reference_count,dimension", [(3, 4, 8), (6, 5, 12), (9, 7, 16)])
def test_low_rank_fid_matches_existing_scipy_sqrtm_for_small_psd_cases(
    generated_count, reference_count, dimension
):
    evaluator = _evaluator_module()
    rng = np.random.default_rng(1000 + generated_count + reference_count + dimension)
    generated = rng.normal(size=(generated_count, dimension))
    reference = rng.normal(size=(reference_count, dimension))

    expected = _scipy_fid(generated, reference)
    actual = evaluator._fid_low_rank(generated, reference)

    assert evaluator._should_use_low_rank_fid(generated, reference)
    assert actual == pytest.approx(expected, rel=2e-7, abs=2e-6)


def _install_fake_fid_module(monkeypatch, value: float) -> None:
    score = types.ModuleType("score")
    score.__path__ = []
    fid = types.ModuleType("score.fid")
    fid.calculate_frechet_distance = lambda *_: value
    monkeypatch.setitem(sys.modules, "score", score)
    monkeypatch.setitem(sys.modules, "score.fid", fid)


def _args(tmp_path: Path, *, mode: str, headline_output: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        repo=tmp_path / "repo",
        metrics_root=tmp_path / "metrics",
        output=tmp_path / "metrics.json",
        headline_output=headline_output,
        samples=None,
        generated_features_cache=None,
        metric_backend="canonical",
        inception_batch_size=16,
        vgg_batch_size=128,
        knn_query_batch=1024,
        kid=False,
        kid_subsets=3,
        kid_subset_size=4,
        kid_seed=77,
        mode=mode,
        per_class_output=None,
        longtail_groups="none",
    )


def test_headline_mode_reuses_one_inception_pass_includes_seeded_kid_and_skips_details(tmp_path, monkeypatch):
    evaluator = _evaluator_module()
    args = _args(tmp_path, mode="headline", headline_output=tmp_path / "headline.json")
    args.metrics_root.mkdir()
    generated_features = np.arange(40, dtype=np.float32).reshape(10, 4)
    reference_features = generated_features + 0.5
    np.save(args.metrics_root / "cifar10_feats.npy", reference_features)
    np.savez(args.metrics_root / "cifar10.train.npz", mu=np.zeros(4), sigma=np.eye(4))
    _install_fake_fid_module(monkeypatch, 12.5)

    inception_calls = []
    monkeypatch.setattr(
        evaluator,
        "_inception_features_and_is",
        lambda repo, images, batch_size, *, device: (
            inception_calls.append((repo, images, batch_size, device))
            or generated_features,
            (8.5, 0.2),
        ),
    )
    monkeypatch.setattr(evaluator, "_standard_prd_scores", lambda generated, reference, nclass, **_: (0.61, 0.42))
    kid_calls = []

    def fake_kid(generated, reference, *, num_subsets, max_subset_size, rng):
        kid_calls.append((generated, reference, num_subsets, max_subset_size, rng.integers(1_000_000)))
        return 0.0125

    monkeypatch.setattr(evaluator, "polynomial_mmd_kid", fake_kid)
    monkeypatch.setattr(
        evaluator,
        "improved_prd_vgg16_k3",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("headline mode must not run VGG improved-PRD")),
    )

    payload = evaluator._evaluate(
        args,
        images=object(),
        labels=np.zeros(10, dtype=np.int64),
        counts=np.array([10] + [0] * 9),
        nclass=10,
    )

    assert len(inception_calls) == 1
    assert kid_calls[0][0] is generated_features
    np.testing.assert_allclose(kid_calls[0][1], reference_features)
    assert kid_calls[0][2:4] == (3, 4)
    assert kid_calls[0][4] == np.random.default_rng(77).integers(1_000_000)
    assert set(payload["metrics"]) == {"FID", "IS", "IS_std", "F_8", "F_1_8", "KID"}
    assert payload["metrics"]["KID"] == 0.0125
    assert "ImprovedPrecision" not in payload["metrics"]
    assert payload["protocol"]["evaluation_mode"] == "headline"
    assert "omits VGG16" in payload["protocol"]["important_caveat"]
    assert json.loads(args.output.read_text(encoding="utf-8")) == payload
    assert json.loads(args.headline_output.read_text(encoding="utf-8")) == payload


def test_quick_main_accepts_small_uniform_sample_without_cuda(tmp_path, monkeypatch, capsys):
    evaluator = _evaluator_module()
    samples = tmp_path / "samples.npy"
    labels_path = tmp_path / "labels.npy"
    images = np.zeros((20, 3, 32, 32), dtype=np.uint8)
    labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    np.save(samples, images)
    np.save(labels_path, labels)
    per_class_output = tmp_path / "per-class.json"
    observed = {}

    def fake_evaluate(args, received_images, received_labels, counts, nclass):
        observed.update({
            "mode": args.mode,
            "images": received_images,
            "labels": received_labels,
            "counts": counts,
            "nclass": nclass,
            "per_class_output": args.per_class_output,
        })
        return {"metrics": {"FID": 1.0}}

    monkeypatch.setattr(evaluator, "_evaluate", fake_evaluate)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_ccua.py",
        "--repo", str(tmp_path / "repo"),
        "--data-type", "cifar10lt",
        "--samples", str(samples),
        "--labels", str(labels_path),
        "--metrics-root", str(tmp_path / "metrics"),
        "--output", str(tmp_path / "metrics.json"),
        "--mode", "quick",
        "--per-class-output", str(per_class_output),
    ])

    evaluator.main()

    assert observed["mode"] == "quick"
    assert observed["images"].shape == (20, 3, 32, 32)
    assert observed["labels"].shape == (20,)
    np.testing.assert_array_equal(observed["counts"], np.full(10, 2))
    assert observed["nclass"] == 10
    assert observed["per_class_output"] == per_class_output
    assert json.loads(capsys.readouterr().out) == {"FID": 1.0}


def test_quick_mode_returns_only_fast_metrics_and_skips_vgg_and_per_class(tmp_path, monkeypatch):
    evaluator = _evaluator_module()
    args = _args(tmp_path, mode="quick")
    args.metrics_root.mkdir()
    generated_features = np.arange(80, dtype=np.float32).reshape(20, 4)
    reference_features = np.arange(160, dtype=np.float32).reshape(40, 4) + 0.5
    reference_labels = np.repeat(np.arange(10, dtype=np.int64), 4)
    np.save(args.metrics_root / "cifar10_feats.npy", reference_features)
    np.save(args.metrics_root / "cifar10_labels.npy", reference_labels)
    np.savez(args.metrics_root / "cifar10.train.npz", mu=np.zeros(4), sigma=np.eye(4))
    _install_fake_fid_module(monkeypatch, 12.5)

    monkeypatch.setattr(
        evaluator,
        "_inception_features_and_is",
        lambda repo, images, batch_size, *, device: (generated_features, (8.5, 0.2)),
    )
    prd_calls = []

    def fake_prd(generated, reference, nclass, **_kwargs):
        prd_calls.append((generated, reference, nclass))
        return 0.61, 0.42

    monkeypatch.setattr(evaluator, "_standard_prd_scores", fake_prd)
    kid_calls = []

    def fake_kid(generated, reference, **_kwargs):
        kid_calls.append((generated, reference))
        return 0.0125

    monkeypatch.setattr(evaluator, "polynomial_mmd_kid", fake_kid)
    monkeypatch.setattr(
        evaluator,
        "improved_prd_vgg16_k3",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("quick mode must not run VGG improved-PRD")),
    )
    monkeypatch.setattr(
        evaluator,
        "_per_class_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("quick mode must not run per-class FID")),
    )

    payload = evaluator._evaluate(
        args,
        images=np.zeros((20, 3, 32, 32), dtype=np.uint8),
        labels=np.repeat(np.arange(10, dtype=np.int64), 2),
        counts=np.full(10, 2),
        nclass=10,
    )

    assert set(payload["metrics"]) == {"FID", "IS", "IS_std", "F_8", "F_1_8", "KID"}
    assert payload["protocol"]["evaluation_mode"] == "quick"
    assert payload["protocol"]["samples"] == 20
    assert "20 total" in payload["protocol"]["prd_reference"]
    expected_indices = np.concatenate([np.arange(class_id * 4, class_id * 4 + 2) for class_id in range(10)])
    np.testing.assert_allclose(prd_calls[0][1], reference_features[expected_indices])
    np.testing.assert_allclose(kid_calls[0][1], reference_features)
    caveat = payload["protocol"]["important_caveat"].lower()
    assert "not for paper" in caveat
    assert "50k" in caveat


def test_quick_mode_writes_dynamic_per_class_output_without_vgg(tmp_path, monkeypatch):
    evaluator = _evaluator_module()
    args = _args(tmp_path, mode="quick")
    args.metrics_root.mkdir()
    args.per_class_output = tmp_path / "per-class.json"
    args.longtail_groups = "cm_three_way"
    generated_features = np.arange(80, dtype=np.float32).reshape(20, 4)
    reference_features = np.arange(160, dtype=np.float32).reshape(40, 4) + 0.5
    reference_labels = np.repeat(np.arange(10, dtype=np.int64), 4)
    np.save(args.metrics_root / "cifar10_feats.npy", reference_features)
    np.save(args.metrics_root / "cifar10_labels.npy", reference_labels)
    np.savez(args.metrics_root / "cifar10.train.npz", mu=np.zeros(4), sigma=np.eye(4))
    _install_fake_fid_module(monkeypatch, 12.5)

    monkeypatch.setattr(
        evaluator,
        "_inception_features_and_is",
        lambda repo, images, batch_size, *, device: (generated_features, (8.5, 0.2)),
    )
    monkeypatch.setattr(evaluator, "_standard_prd_scores", lambda generated, reference, nclass, **_: (0.61, 0.42))
    monkeypatch.setattr(evaluator, "polynomial_mmd_kid", lambda *args, **kwargs: 0.0125)
    monkeypatch.setattr(
        evaluator,
        "improved_prd_vgg16_k3",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("quick mode must not run VGG improved-PRD")),
    )

    payload = evaluator._evaluate(
        args,
        images=np.zeros((20, 3, 32, 32), dtype=np.uint8),
        labels=np.repeat(np.arange(10, dtype=np.int64), 2),
        counts=np.full(10, 2),
        nclass=10,
    )

    per_class_payload = json.loads(args.per_class_output.read_text(encoding="utf-8"))
    assert set(payload["metrics"]) == {"FID", "IS", "IS_std", "F_8", "F_1_8", "KID"}
    assert per_class_payload["protocol"]["sample_count"] == 20
    assert per_class_payload["protocol"]["sample_schedule"] == "same 20 class-uniform samples as overall table"
    assert "class-uniform 20 sample" in per_class_payload["protocol"]["important_caveat"]
    assert len(per_class_payload["per_class"]) == 10
    assert set(per_class_payload["groups"]) == {"Many", "Medium", "Few"}


@pytest.mark.parametrize("mode", ("detailed", "headline"))
def test_paper_modes_reject_short_samples(tmp_path, monkeypatch, mode):
    evaluator = _evaluator_module()
    samples = tmp_path / "samples.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(samples, np.zeros((20, 3, 32, 32), dtype=np.uint8))
    np.save(labels_path, np.repeat(np.arange(10, dtype=np.int64), 2))
    monkeypatch.setattr(evaluator, "_evaluate", lambda *_args, **_kwargs: pytest.fail("paper mode must fail before evaluation"))
    monkeypatch.setattr(sys, "argv", [
        "evaluate_ccua.py",
        "--repo", str(tmp_path / "repo"),
        "--data-type", "cifar10lt",
        "--samples", str(samples),
        "--labels", str(labels_path),
        "--metrics-root", str(tmp_path / "metrics"),
        "--output", str(tmp_path / "metrics.json"),
        "--mode", mode,
    ])

    with pytest.raises(ValueError, match="paper protocol requires 50k"):
        evaluator.main()


def test_headline_mode_rejects_per_class_output_before_cuda_or_metric_loading(tmp_path, monkeypatch, capsys):
    evaluator = _evaluator_module()
    monkeypatch.setattr(sys, "argv", [
        "evaluate_ccua.py",
        "--repo", str(tmp_path / "repo"),
        "--data-type", "cifar10lt",
        "--samples", str(tmp_path / "samples.npy"),
        "--labels", str(tmp_path / "labels.npy"),
        "--metrics-root", str(tmp_path / "metrics"),
        "--output", str(tmp_path / "metrics.json"),
        "--mode", "headline",
        "--per-class-output", str(tmp_path / "per-class.json"),
    ])

    with pytest.raises(SystemExit) as error:
        evaluator.main()

    assert error.value.code == 2
    assert "intentionally omits per-class FID" in capsys.readouterr().err


def test_detailed_mode_publishes_atomic_headline_before_vgg_and_keeps_it_separate(tmp_path, monkeypatch):
    evaluator = _evaluator_module()
    headline_output = tmp_path / "headline.json"
    args = _args(tmp_path, mode="detailed", headline_output=headline_output)
    args.metrics_root.mkdir()
    headline_metrics = {"FID": 1.0, "IS": 2.0, "IS_std": 0.1, "F_8": 0.3, "F_1_8": 0.4}
    monkeypatch.setattr(
        evaluator,
        "_collect_headline_metrics",
        lambda *_: (headline_metrics, np.zeros((2, 4), dtype=np.float32), np.zeros((2, 4), dtype=np.float32)),
    )
    np.save(args.metrics_root / "cifar10_vgg16_fc2.npy", np.zeros((2, 4), dtype=np.float32))
    np.save(args.metrics_root / "cifar10_vgg16_fc2_k3_radii.npy", np.zeros(2, dtype=np.float32))

    def fake_improved_prd(*_args, **_kwargs):
        early_primary = json.loads(args.output.read_text(encoding="utf-8"))
        early_separate = json.loads(headline_output.read_text(encoding="utf-8"))
        assert early_primary == early_separate
        assert early_primary["protocol"]["evaluation_mode"] == "headline"
        assert "ImprovedPrecision" not in early_primary["metrics"]
        return 0.7, 0.8

    monkeypatch.setattr(evaluator, "improved_prd_vgg16_k3", fake_improved_prd)

    payload = evaluator._evaluate(
        args,
        images=np.zeros((2, 3, 32, 32), dtype=np.uint8),
        labels=np.zeros(2, dtype=np.int64),
        counts=np.array([2] + [0] * 9),
        nclass=10,
    )

    assert payload["metrics"]["ImprovedPrecision"] == 0.7
    assert payload["metrics"]["Recall"] == 0.8
    assert "evaluation_mode" not in payload["protocol"]
    assert "ImprovedPrecision" not in json.loads(headline_output.read_text(encoding="utf-8"))["metrics"]


def test_generated_inception_cache_is_reused_only_for_the_same_sample_artifact(tmp_path, monkeypatch):
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
    first = evaluator._cached_inception_features_and_is(args, np.zeros((3, 3, 32, 32), dtype=np.uint8))
    second = evaluator._cached_inception_features_and_is(args, np.zeros((3, 3, 32, 32), dtype=np.uint8))

    assert len(calls) == 1
    np.testing.assert_array_equal(first[0], features)
    np.testing.assert_array_equal(second[0], features)
    assert second[1] == (4.0, 0.2)


def test_gpu_fast_mode_publishes_preview_and_keeps_split_fids_out_of_canonical_names(tmp_path, monkeypatch):
    evaluator = _evaluator_module()
    args = _args(tmp_path, mode="gpu_fast")
    args.per_class_output = tmp_path / "fast_per_class.json"
    fast_metrics = {"FID": 8.0, "IS": 7.0, "IS_std": 0.1, "KID": 0.02}
    generated = np.zeros((20, 4), dtype=np.float32)
    reference = np.zeros((40, 4), dtype=np.float32)
    monkeypatch.setattr(
        evaluator,
        "_collect_gpu_fast_metrics",
        lambda *_: (dict(fast_metrics), generated, reference),
    )
    monkeypatch.setattr(
        evaluator,
        "_per_class_payload",
        lambda *_: {"groups": {"Many": {"FID": 1.0}, "Medium": {"FID": 2.0}, "Few": {"FID": 3.0}}},
    )

    payload = evaluator._evaluate(
        args,
        images=np.zeros((20, 3, 32, 32), dtype=np.uint8),
        labels=np.repeat(np.arange(10, dtype=np.int64), 2),
        counts=np.full(10, 2),
        nclass=10,
    )

    assert payload["protocol"]["evaluation_mode"] == "gpu_fast"
    assert payload["protocol"]["paper_report_ready"] is False
    assert payload["metrics"]["Few_FID"] == 3.0
    assert payload["protocol"]["omitted_metrics"] == ["F_8", "F_1_8", "ImprovedPrecision", "Recall"]
    assert json.loads(args.output.read_text(encoding="utf-8")) == payload
    assert args.per_class_output.exists()


def test_improved_pr_mode_skips_inception_prd_and_writes_only_precision_recall(tmp_path, monkeypatch):
    evaluator = _evaluator_module()
    args = _args(tmp_path, mode="improved_pr")
    args.metrics_root.mkdir()
    np.save(args.metrics_root / "cifar10_vgg16_fc2.npy", np.zeros((40, 6), dtype=np.float32))
    np.save(args.metrics_root / "cifar10_vgg16_fc2_k3_radii.npy", np.ones(40, dtype=np.float32))

    observed = {}

    def fake_improved(
        images, reference_features, reference_radii, *, batch_size, query_batch, device
    ):
        observed.update({
            "images": images,
            "reference_features": reference_features,
            "reference_radii": reference_radii,
            "batch_size": batch_size,
            "query_batch": query_batch,
            "device": device,
        })
        return 0.71, 0.43

    monkeypatch.setattr(evaluator, "improved_prd_vgg16_k3", fake_improved)
    monkeypatch.setattr(
        evaluator,
        "_cached_inception_features_and_is",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("improved_pr must skip Inception")),
    )

    payload = evaluator._evaluate(
        args,
        images=np.zeros((20, 3, 32, 32), dtype=np.uint8),
        labels=np.repeat(np.arange(10, dtype=np.int64), 2),
        counts=np.full(10, 2),
        nclass=10,
    )

    assert payload["metrics"] == {"ImprovedPrecision": 0.71, "Recall": 0.43}
    assert payload["protocol"]["evaluation_mode"] == "improved_pr_only"
    assert payload["protocol"]["reported_metrics"] == ["ImprovedPrecision", "Recall"]
    assert "F_8" in payload["protocol"]["omitted_metrics"]
    assert json.loads(args.output.read_text(encoding="utf-8")) == payload
    assert observed["batch_size"] == 128
    assert observed["query_batch"] == 1024
    assert observed["device"] == "cuda"
