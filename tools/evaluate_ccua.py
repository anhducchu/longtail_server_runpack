#!/usr/bin/env python3
"""Evaluate one generated CIFAR array with a pinned common metric protocol.

``--mode headline`` is the early-result protocol: it reports the common
Inception FID/IS/PRD columns and deterministic KID, but deliberately omits
the exact VGG16 50k-by-50k improved-PRD and every per-class FID. The default
``detailed`` mode writes the canonical final metric set and an atomic
headline snapshot before starting those optional expensive calculations.
``--mode quick`` accepts a smaller class-uniform smoke sample and reports only
the common fast metrics; its values are not for paper reporting or direct
comparison with the 50k protocol. ``--mode gpu_fast`` is a deliberately
non-canonical 50k preview: it uses CUDA for FID/KID and per-class/split FID,
omits the CPU Inception PRD and VGG improved-PRD columns, and is never used by
the paper report. ``--mode improved_pr`` computes only the canonical VGG16
improved Precision/Recall pair, reusing the balanced reference features and
radii; it intentionally skips Inception PRD/F8/F1/8 and all other metrics.
All modes can reuse the generated Inception feature cache.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from ltx.paper_metrics import improved_prd_vgg16_k3, polynomial_mmd_kid, polynomial_mmd_kid_gpu


_IS_SPLITS = 10


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON without exposing a partially written metric artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _feature_cache_signature(samples: Path) -> dict:
    """Return a cheap identity for a generated sample artifact.

    The cache is intentionally tied to path, byte size, and mtime.  A full
    SHA-256 over a 50k image array would make a fast re-evaluation needlessly
    expensive. The run directory and task-resolved metadata retain checkpoint
    and protocol identity.
    """
    resolved = Path(samples).resolve()
    stat = resolved.stat()
    return {
        "sample_path": str(resolved),
        "sample_size": int(stat.st_size),
        "sample_mtime_ns": int(stat.st_mtime_ns),
    }


def _load_feature_cache(path: Path, samples: Path, num_images: int) -> tuple[np.ndarray, tuple[float, float]] | None:
    """Load a validated generated Inception cache, returning ``None`` on miss."""
    if not path.is_file():
        return None
    try:
        signature = _feature_cache_signature(samples)
        with np.load(path, allow_pickle=False) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)
            cached_signature = {
                "sample_path": str(np.asarray(payload["sample_path"]).item()),
                "sample_size": int(np.asarray(payload["sample_size"]).item()),
                "sample_mtime_ns": int(np.asarray(payload["sample_mtime_ns"]).item()),
            }
            is_score = float(np.asarray(payload["is_score"]).item())
            is_std = float(np.asarray(payload["is_std"]).item())
        if cached_signature != signature:
            print(f"[metrics] Inception cache stale, recomputing: {path}", file=sys.stderr)
            return None
        if features.shape != (num_images, 2048):
            print(f"[metrics] Inception cache shape mismatch, recomputing: {path}", file=sys.stderr)
            return None
        if not np.isfinite(features).all() or not np.isfinite([is_score, is_std]).all():
            print(f"[metrics] Inception cache contains non-finite values, recomputing: {path}", file=sys.stderr)
            return None
        print(f"[metrics] reusing generated Inception features: {path}", file=sys.stderr)
        return features, (is_score, is_std)
    except (OSError, KeyError, ValueError, TypeError, EOFError) as exc:
        print(f"[metrics] invalid Inception cache ({exc}), recomputing: {path}", file=sys.stderr)
        return None


def _atomic_write_feature_cache(path: Path, samples: Path, features: np.ndarray, is_score: float, is_std: float) -> None:
    """Atomically publish an uncompressed feature cache beside the samples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = _feature_cache_signature(samples)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(
                handle,
                features=np.asarray(features, dtype=np.float32),
                is_score=np.asarray(is_score, dtype=np.float64),
                is_std=np.asarray(is_std, dtype=np.float64),
                sample_path=np.asarray(signature["sample_path"]),
                sample_size=np.asarray(signature["sample_size"], dtype=np.int64),
                sample_mtime_ns=np.asarray(signature["sample_mtime_ns"], dtype=np.int64),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _inception_features_and_is(
    repo: Path, images: np.ndarray, batch_size: int, device: str = "cuda"
) -> tuple[np.ndarray, tuple[float, float]]:
    """Extract pinned 2048-d Inception features once and calculate IS online.

    The released ``score.both`` helper hides the activation array after it
    computes FID/IS/PRD. Keeping that array here lets KID and optional
    class-level FIDs reuse the same model pass. IS is accumulated by split,
    avoiding a second 50k-by-1008 host array while preserving its formula.
    """
    import torch

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA Inception extraction requested, but CUDA is unavailable")
    sys.path.insert(0, str(repo))
    from score.inception import InceptionV3

    num_images = len(images)
    feature_dim = 2048
    model = InceptionV3([
        InceptionV3.BLOCK_INDEX_BY_DIM[feature_dim],
        InceptionV3.BLOCK_INDEX_BY_DIM["prob"],
    ]).to(target_device).eval()
    features = np.empty((num_images, feature_dim), dtype=np.float32)
    split_sizes = np.array(
        [((index + 1) * num_images // _IS_SPLITS) - (index * num_images // _IS_SPLITS)
         for index in range(_IS_SPLITS)],
        dtype=np.int64,
    )
    probability_sums: np.ndarray | None = None
    probability_log_sums = np.zeros(_IS_SPLITS, dtype=np.float64)

    with torch.no_grad():
        for start in range(0, num_images, batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(images[start:start + batch_size])).float().to(
                target_device, non_blocking=target_device.type == "cuda"
            )
            end = start + len(batch)
            prediction = model(batch)
            features[start:end] = prediction[0].flatten(1).cpu().numpy().astype(np.float32, copy=False)
            probabilities = prediction[1].cpu().numpy().astype(np.float64, copy=False)
            if probability_sums is None:
                probability_sums = np.zeros((_IS_SPLITS, probabilities.shape[1]), dtype=np.float64)

            cursor = start
            while cursor < end:
                split = min(_IS_SPLITS - 1, cursor * _IS_SPLITS // num_images)
                split_end = min(end, (split + 1) * num_images // _IS_SPLITS)
                chunk = probabilities[cursor - start:split_end - start]
                probability_sums[split] += np.sum(chunk, axis=0, dtype=np.float64)
                probability_log_sums[split] += np.sum(chunk * np.log(chunk), dtype=np.float64)
                cursor = split_end

    if probability_sums is None:
        raise ValueError("Inception Score requires at least one image")
    scores = []
    for split, split_size in enumerate(split_sizes):
        if split_size <= 0:
            raise ValueError(f"Inception Score requires at least {_IS_SPLITS} images")
        average_probability = probability_sums[split] / split_size
        scores.append(np.exp(
            probability_log_sums[split] / split_size
            - np.sum(average_probability * np.log(average_probability), dtype=np.float64)
        ))
    return features, (float(np.mean(scores)), float(np.std(scores)))


def _cached_inception_features_and_is(args, images: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Extract Inception features once per sample artifact and reuse them.

    The cache is opt-in at the evaluator API level and enabled by the shared
    CCUA adapter for subsequent campaigns.  A missing or stale cache simply
    falls back to the normal CUDA extraction path and is written atomically.
    """
    cache_value = getattr(args, "generated_features_cache", None)
    samples_value = getattr(args, "samples", None)
    cache = Path(cache_value) if cache_value else None
    samples = Path(samples_value) if samples_value else None
    if cache is not None and samples is not None:
        cached = _load_feature_cache(cache, samples, len(images))
        if cached is not None:
            return cached

    result = _inception_features_and_is(
        args.repo,
        images,
        batch_size=args.inception_batch_size,
        device=getattr(args, "device", "cuda"),
    )
    if cache is not None and samples is not None:
        _atomic_write_feature_cache(cache, samples, result[0], result[1][0], result[1][1])
        print(f"[metrics] cached generated Inception features: {cache}", file=sys.stderr)
    return result


def _validate_feature_pair(features: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    generated = np.asarray(features)
    real = np.asarray(reference)
    if generated.ndim != 2 or real.ndim != 2 or generated.shape[1] != real.shape[1]:
        raise ValueError(
            "FID expects generated/reference arrays with matching [N, feature_dim] shapes, got "
            f"{generated.shape} and {real.shape}"
        )
    if len(generated) < 2 or len(real) < 2:
        raise ValueError(f"FID requires at least two samples on each side, got {len(generated)} and {len(real)}")
    return generated, real


def _fid_low_rank(features: np.ndarray, reference: np.ndarray) -> float:
    """Exact sample-covariance FID from the nuclear norm of a small Gram matrix.

    For centred feature matrices A and B scaled by ``sqrt(n - 1)``, their
    sample covariances are A.T @ A and B.T @ B. Therefore the covariance cross
    term is ``||A @ B.T||_*``. This is the same Frechet expression as SciPy's
    ``sqrtm(C1 @ C2)`` but, for small class slices, SVDs an
    ``n_generated`` by ``n_reference`` matrix instead of a 2048 by 2048 one.
    """
    generated, real = _validate_feature_pair(features, reference)
    generated = np.asarray(generated, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    generated_mean = np.mean(generated, axis=0, dtype=np.float64)
    real_mean = np.mean(real, axis=0, dtype=np.float64)
    generated_factor = (generated - generated_mean) / np.sqrt(len(generated) - 1)
    real_factor = (real - real_mean) / np.sqrt(len(real) - 1)
    covariance_trace = (
        np.sum(generated_factor * generated_factor, dtype=np.float64)
        + np.sum(real_factor * real_factor, dtype=np.float64)
    )
    covariance_cross_trace = np.sum(
        np.linalg.svd(generated_factor @ real_factor.T, compute_uv=False), dtype=np.float64
    )
    mean_distance = np.sum((generated_mean - real_mean) ** 2, dtype=np.float64)
    return float(mean_distance + covariance_trace - 2.0 * covariance_cross_trace)


def _should_use_low_rank_fid(features: np.ndarray, reference: np.ndarray) -> bool:
    """Use the Gram formulation only when its dense SVD is actually smaller."""
    generated, real = _validate_feature_pair(features, reference)
    dimension = generated.shape[1]
    return len(generated) * len(real) <= dimension * dimension


def _fid(features: np.ndarray, reference: np.ndarray, calculate_frechet_distance) -> float:
    generated, real = _validate_feature_pair(features, reference)
    if _should_use_low_rank_fid(generated, real):
        return _fid_low_rank(generated, real)
    return float(calculate_frechet_distance(
        np.mean(generated, axis=0, dtype=np.float64), np.cov(generated, rowvar=False),
        np.mean(real, axis=0, dtype=np.float64), np.cov(real, rowvar=False),
    ))


def _fid_to_statistics(features: np.ndarray, reference_mean: np.ndarray, reference_covariance: np.ndarray,
                       calculate_frechet_distance) -> float:
    generated = np.asarray(features)
    if generated.ndim != 2 or len(generated) < 2:
        raise ValueError(f"FID requires at least two generated feature vectors, got {generated.shape}")
    return float(calculate_frechet_distance(
        np.mean(generated, axis=0, dtype=np.float64), np.cov(generated, rowvar=False),
        reference_mean, reference_covariance,
    ))


def _require_cuda_torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU metric preview requires CUDA")
    return torch, torch.device("cuda", torch.cuda.current_device())


def _frechet_from_statistics_gpu(
    generated_mean: np.ndarray,
    generated_covariance: np.ndarray,
    reference_mean: np.ndarray,
    reference_covariance: np.ndarray,
) -> float:
    """Evaluate the standard sample-covariance FID formula on CUDA.

    The PSD eigensolver form is numerically stable for covariance matrices and
    differs from the release SciPy ``sqrtm`` only by floating-point/backend
    round-off.  It is used by ``gpu_fast`` only; canonical paper FID continues
    to call the pinned release implementation.
    """
    torch, device = _require_cuda_torch()
    mu_generated = torch.as_tensor(np.asarray(generated_mean, dtype=np.float64), device=device)
    cov_generated = torch.as_tensor(
        np.asarray(generated_covariance, dtype=np.float64), device=device
    )
    mu_reference = torch.as_tensor(np.asarray(reference_mean, dtype=np.float64), device=device)
    cov_reference = torch.as_tensor(
        np.asarray(reference_covariance, dtype=np.float64), device=device
    )
    cov_generated = (cov_generated + cov_generated.T) * 0.5
    cov_reference = (cov_reference + cov_reference.T) * 0.5
    eigvals, eigvecs = torch.linalg.eigh(cov_generated)
    sqrt_cov_generated = (
        eigvecs * eigvals.clamp_min(0).sqrt().unsqueeze(0)
    ) @ eigvecs.T
    middle = sqrt_cov_generated @ cov_reference @ sqrt_cov_generated
    middle = (middle + middle.T) * 0.5
    middle_eigvals = torch.linalg.eigvalsh(middle).clamp_min(0)
    mean_distance = (mu_generated - mu_reference).square().sum()
    value = (
        mean_distance
        + torch.trace(cov_generated)
        + torch.trace(cov_reference)
        - 2.0 * middle_eigvals.sqrt().sum()
    )
    return float(value.item())


def _fid_to_statistics_gpu(
    features: np.ndarray,
    reference_mean: np.ndarray,
    reference_covariance: np.ndarray,
) -> float:
    generated = np.asarray(features)
    if generated.ndim != 2 or len(generated) < 2:
        raise ValueError(f"FID requires at least two generated feature vectors, got {generated.shape}")
    torch, device = _require_cuda_torch()
    generated_gpu = torch.as_tensor(
        np.ascontiguousarray(generated, dtype=np.float64), device=device
    )
    generated_mean = generated_gpu.mean(dim=0)
    centered = generated_gpu - generated_mean
    generated_covariance = centered.T @ centered / (len(generated) - 1)
    return _frechet_from_statistics_gpu(
        generated_mean.detach().cpu().numpy(),
        generated_covariance.detach().cpu().numpy(),
        reference_mean,
        reference_covariance,
    )


def _fid_low_rank_gpu(features: np.ndarray, reference: np.ndarray) -> float:
    """CUDA version of the exact low-rank sample-covariance FID identity."""
    generated, real = _validate_feature_pair(features, reference)
    torch, device = _require_cuda_torch()
    generated_gpu = torch.as_tensor(
        np.ascontiguousarray(generated, dtype=np.float64), device=device
    )
    real_gpu = torch.as_tensor(np.ascontiguousarray(real, dtype=np.float64), device=device)
    generated_mean = generated_gpu.mean(dim=0)
    real_mean = real_gpu.mean(dim=0)
    generated_factor = (generated_gpu - generated_mean) / np.sqrt(len(generated) - 1)
    real_factor = (real_gpu - real_mean) / np.sqrt(len(real) - 1)
    covariance_trace = generated_factor.square().sum() + real_factor.square().sum()
    covariance_cross_trace = torch.linalg.svdvals(generated_factor @ real_factor.T).sum()
    mean_distance = (generated_mean - real_mean).square().sum()
    return float((mean_distance + covariance_trace - 2.0 * covariance_cross_trace).item())


def _fid_gpu(features: np.ndarray, reference: np.ndarray) -> float:
    """Use the same FID definition with a CUDA low-rank/eigensolver backend."""
    generated, real = _validate_feature_pair(features, reference)
    if _should_use_low_rank_fid(generated, real):
        return _fid_low_rank_gpu(generated, real)
    generated = np.asarray(generated, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    generated_mean = np.mean(generated, axis=0, dtype=np.float64)
    real_mean = np.mean(real, axis=0, dtype=np.float64)
    generated_centered = generated - generated_mean
    real_centered = real - real_mean
    generated_covariance = generated_centered.T @ generated_centered / (len(generated) - 1)
    real_covariance = real_centered.T @ real_centered / (len(real) - 1)
    return _frechet_from_statistics_gpu(
        generated_mean, generated_covariance, real_mean, real_covariance
    )


def _standard_prd_scores(generated_features: np.ndarray, reference_features: np.ndarray,
                         nclass: int) -> tuple[float, float]:
    """Run the canonical 10-run Inception PRD protocol on reused activations."""
    from score.prd_score import compute_prd_from_embedding, prd_to_max_f_beta_pair

    prd_data = compute_prd_from_embedding(
        eval_data=generated_features,
        ref_data=reference_features,
        num_clusters=nclass * 20,
        num_angles=1001,
        num_runs=10,
        enforce_balance=True,
    )
    f_8, f_1_8 = prd_to_max_f_beta_pair(prd_data[0], prd_data[1], beta=8)
    return float(f_8), float(f_1_8)


def _three_way_groups(nclass: int) -> dict[str, np.ndarray]:
    if nclass == 10:
        return {
            "Many": np.arange(0, 3),
            "Medium": np.arange(3, 6),
            "Few": np.arange(6, 10),
        }
    if nclass == 100:
        return {
            "Many": np.arange(0, 33),
            "Medium": np.arange(33, 66),
            "Few": np.arange(66, 100),
        }
    raise ValueError(f"Many/Medium/Few grouping only supports 10 or 100 classes, got {nclass}")


def _base_protocol(nclass: int, args, sample_count: int = 50000) -> dict:
    return {
        "samples": sample_count,
        "labels": "uniform support across classes",
        "real_reference": "balanced CIFAR train",
        "standard_prd": f"InceptionV3, {nclass * 20} clusters",
        "improved_prd": "VGG16 fc2, exact k-NN manifold k=3",
        "inception_batch_size": args.inception_batch_size,
        "inception_device": getattr(args, "device", "cuda"),
    }


def _headline_protocol(protocol: dict, mode: str = "headline") -> dict:
    headline = dict(protocol)
    if mode == "quick":
        caveat = (
            f"Quick smoke uses {protocol['samples']} class-uniform samples instead of the paper's 50k; "
            "it is not for paper reporting and must not be compared directly with 50k results. "
            f"{protocol['prd_reference']}. "
            "It omits VGG16 fc2 exact improved-PRD (ImprovedPrecision/Recall) "
            "from the overall quick report; optional per-class FIDs are written separately."
        )
    else:
        caveat = (
            "Headline mode omits VGG16 fc2 exact improved-PRD (ImprovedPrecision/Recall) "
            "and all per-class/split FIDs. F_8 and F_1_8 remain the standard Inception PRD protocol."
        )
    headline.update({"evaluation_mode": mode, "important_caveat": caveat})
    return headline


def _quick_prd_reference_subset(
    args, reference_features: np.ndarray, nclass: int, sample_count: int, dataset: str
) -> np.ndarray:
    """Select the deterministic balanced real subset used by quick PRD."""
    labels_path = args.metrics_root / f"{dataset}_labels.npy"
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"quick PRD requires the balanced-reference labels asset {labels_path}; "
            "rerun tools/prepare_cifar_metric_assets.py"
        )
    reference_labels = np.load(labels_path, mmap_mode="r")
    if reference_labels.ndim != 1 or reference_labels.shape[0] != reference_features.shape[0]:
        raise ValueError(
            "quick PRD requires reference features and labels with matching [N] lengths, got "
            f"features={reference_features.shape} labels={reference_labels.shape}"
        )
    per_class, remainder = divmod(sample_count, nclass)
    if remainder:
        raise ValueError(
            f"quick PRD requires a sample count divisible by {nclass}, got N={sample_count}"
        )
    selected = []
    reference_labels_array = np.asarray(reference_labels)
    for class_id in range(nclass):
        class_indices = np.flatnonzero(reference_labels_array == class_id)
        if len(class_indices) < per_class:
            raise ValueError(
                f"quick PRD reference has only {len(class_indices)} samples for class {class_id}; "
                f"requires {per_class}"
            )
        selected.append(class_indices[:per_class])
    return np.asarray(reference_features[np.concatenate(selected)])


def _collect_headline_metrics(args, images: np.ndarray, nclass: int, dataset: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Compute every shared canonical headline metric from one feature array."""
    generated_features, (is_score, is_std) = _cached_inception_features_and_is(args, images)
    reference_features = np.load(args.metrics_root / f"{dataset}_feats.npy", mmap_mode="r")
    prd_reference_features = reference_features
    if args.mode == "quick":
        prd_reference_features = _quick_prd_reference_subset(
            args, reference_features, nclass, len(generated_features), dataset
        )
    sys.path.insert(0, str(args.repo))
    from score.fid import calculate_frechet_distance

    with np.load(args.metrics_root / f"{dataset}.train.npz") as statistics:
        fid = _fid_to_statistics(
            generated_features, statistics["mu"][:], statistics["sigma"][:], calculate_frechet_distance
        )
    f_8, f_1_8 = _standard_prd_scores(
        generated_features,
        prd_reference_features,
        nclass,
    )
    metrics = {
        "FID": float(fid),
        "IS": float(is_score),
        "IS_std": float(is_std),
        "F_8": float(f_8),
        "F_1_8": float(f_1_8),
    }
    if args.kid or args.mode in ("headline", "quick"):
        metrics["KID"] = polynomial_mmd_kid(
            generated_features,
            reference_features,
            num_subsets=args.kid_subsets,
            max_subset_size=args.kid_subset_size,
            rng=np.random.default_rng(args.kid_seed),
        )
    return metrics, generated_features, reference_features


def _collect_gpu_fast_metrics(
    args, images: np.ndarray, nclass: int, dataset: str
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Compute the fast 50k preview without the CPU Inception PRD pass.

    Inception extraction is shared with the canonical phase through the cache.
    FID uses the same sample-covariance definition on a CUDA eigensolver, KID
    uses the same seeded subsets with CUDA Gram matrices, and VGG improved-PRD
    remains a separate canonical phase.  The returned payload is explicitly
    marked as preview-only by :func:`_evaluate`.
    """
    generated_features, (is_score, is_std) = _cached_inception_features_and_is(args, images)
    reference_features = np.load(args.metrics_root / f"{dataset}_feats.npy", mmap_mode="r")
    with np.load(args.metrics_root / f"{dataset}.train.npz") as statistics:
        fid = _fid_to_statistics_gpu(
            generated_features,
            statistics["mu"][:],
            statistics["sigma"][:],
        )
    metrics = {
        "FID": float(fid),
        "IS": float(is_score),
        "IS_std": float(is_std),
        "KID": polynomial_mmd_kid_gpu(
            generated_features,
            reference_features,
            num_subsets=args.kid_subsets,
            max_subset_size=args.kid_subset_size,
            rng=np.random.default_rng(args.kid_seed),
        ),
    }
    return metrics, generated_features, reference_features


def _evaluate_improved_pr_only(
    args, images: np.ndarray, labels: np.ndarray, counts: np.ndarray, nclass: int, dataset: str
) -> dict:
    """Compute only the canonical VGG16 improved Precision/Recall metrics.

    This partial evaluator keeps the detailed mode's VGG16 fc2 and k=3
    manifold definition, but skips the expensive CPU Inception PRD pass and
    does not write a misleading full metric artifact under
    ``metrics.native.json``.
    """
    reference_vgg_features = np.load(args.metrics_root / f"{dataset}_vgg16_fc2.npy", mmap_mode="r")
    reference_radii = np.load(args.metrics_root / f"{dataset}_vgg16_fc2_k3_radii.npy", mmap_mode="r")
    precision, recall = improved_prd_vgg16_k3(
        np.asarray(images),
        np.asarray(reference_vgg_features),
        np.asarray(reference_radii),
        batch_size=args.vgg_batch_size,
        query_batch=args.knn_query_batch,
        device=getattr(args, "device", "cuda"),
    )
    protocol = _base_protocol(nclass, args, sample_count=int(images.shape[0]))
    protocol.update({
        "evaluation_mode": "improved_pr_only",
        "paper_report_ready": False,
        "metric_backend": f"{getattr(args, 'device', 'cuda')}_vgg_exact",
        "reported_metrics": ["ImprovedPrecision", "Recall"],
        "omitted_metrics": ["FID", "IS", "F_8", "F_1_8", "KID", "sFID"],
        "important_caveat": (
            "Partial VGG16 fc2 exact k-NN manifold evaluation. A 10000-sample run is a "
            "within-campaign screening result, not paper-reporting quality. Inception PRD "
            "F_8/F_1_8 and all other metrics were intentionally skipped."
        ),
    })
    payload = {
        "metrics": {
            "ImprovedPrecision": float(precision),
            "Recall": float(recall),
        },
        "protocol": protocol,
        "label_histogram": counts.tolist(),
    }
    _atomic_write_json(args.output, payload)
    return payload


def _per_class_payload(args, generated_features: np.ndarray, reference_features: np.ndarray,
                       labels: np.ndarray, nclass: int, dataset: str) -> dict:
    labels_path = args.metrics_root / f"{dataset}_labels.npy"
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"per-class FID requires the balanced-reference labels asset {labels_path}; "
            "rerun tools/prepare_cifar_metric_assets.py"
        )
    reference_labels = np.load(labels_path, mmap_mode="r")
    sys.path.insert(0, str(args.repo))
    from score.fid import calculate_frechet_distance
    fid_backend = getattr(args, "metric_backend", "canonical")
    if fid_backend not in {"canonical", "gpu"}:
        raise ValueError(f"unknown FID backend {fid_backend!r}")

    def compute_fid(generated_slice: np.ndarray, reference_slice: np.ndarray) -> float:
        if fid_backend == "gpu":
            return _fid_gpu(generated_slice, reference_slice)
        return _fid(generated_slice, reference_slice, calculate_frechet_distance)

    per_class = {}
    for class_id in range(nclass):
        generated_mask = labels == class_id
        reference_mask = np.asarray(reference_labels) == class_id
        generated_slice = generated_features[generated_mask]
        reference_slice = reference_features[reference_mask]
        per_class[str(class_id)] = {
            "FID": compute_fid(generated_slice, reference_slice),
            "generated": int(generated_mask.sum()),
            "reference": int(reference_mask.sum()),
        }
    groups = {}
    if args.longtail_groups == "cm_three_way":
        for name, classes in _three_way_groups(nclass).items():
            generated_mask = np.isin(labels, classes)
            reference_mask = np.isin(reference_labels, classes)
            groups[name] = {
                "FID": compute_fid(generated_features[generated_mask], reference_features[reference_mask]),
                "classes": classes.tolist(),
                "generated": int(generated_mask.sum()),
                "reference": int(reference_mask.sum()),
            }
    sample_count = int(labels.shape[0])
    return {
        "protocol": {
            "feature_extractor": "pinned active-backbone InceptionV3 (2048-d)",
            "reference": "balanced CIFAR train",
            "sample_count": sample_count,
            "sample_schedule": (
                f"same {'50k' if sample_count == 50000 else sample_count} "
                "class-uniform samples as overall table"
            ),
            "fid_computation": (
                "exact sample-covariance FID; class slices whose generated-by-reference Gram matrix "
                "is no larger than 2048-by-2048 use an equivalent low-rank nuclear-norm formulation, "
                "while larger slices retain the release SciPy sqrtm formulation"
                if fid_backend == "canonical" else
                "same sample-covariance FID definition evaluated with CUDA float64 low-rank/eigensolver "
                "kernels; use the canonical evaluator for paper values"
            ),
            "important_caveat": (
                "Published split FIDs may use separately sampled 20k generated images per split. "
                f"These values use the common table's class-uniform "
                f"{'50k' if sample_count == 50000 else sample_count} sample and are only comparable within this campaign."
            ),
        },
        "per_class": per_class,
        "groups": groups,
    }


def _evaluate(args, images: np.ndarray, labels: np.ndarray, counts: np.ndarray, nclass: int) -> dict:
    """Run one protocol, publishing fast previews separately from paper output."""
    dataset = "cifar100" if nclass == 100 else "cifar10"
    if args.mode == "improved_pr":
        return _evaluate_improved_pr_only(args, images, labels, counts, nclass, dataset)
    if args.mode == "gpu_fast":
        args.metric_backend = "gpu"
        metrics, generated_features, reference_features = _collect_gpu_fast_metrics(
            args, images, nclass, dataset
        )
        protocol = _base_protocol(nclass, args, sample_count=int(images.shape[0]))
        protocol.update({
            "evaluation_mode": "gpu_fast",
            "paper_report_ready": False,
            "metric_backend": "cuda_preview",
            "omitted_metrics": ["F_8", "F_1_8", "ImprovedPrecision", "Recall"],
            "important_caveat": (
                "Preview only: FID/KID and optional per-class/split FIDs use CUDA "
                "kernels with the same mathematical definitions, while canonical "
                "paper PRD and VGG improved-PRD are intentionally omitted."
            ),
        })
        if args.per_class_output is not None:
            per_class_payload = _per_class_payload(
                args, generated_features, reference_features, labels, nclass, dataset
            )
            for name, group in per_class_payload["groups"].items():
                metrics[f"{name}_FID"] = float(group["FID"])
            _atomic_write_json(args.per_class_output, per_class_payload)
        payload = {
            "metrics": metrics,
            "protocol": protocol,
            "label_histogram": counts.tolist(),
        }
        _atomic_write_json(args.output, payload)
        return payload

    headline_metrics, generated_features, reference_features = _collect_headline_metrics(args, images, nclass, dataset)
    sample_count = int(images.shape[0]) if args.mode == "quick" else 50000
    protocol = _base_protocol(nclass, args, sample_count=sample_count)
    if args.mode == "quick":
        protocol["prd_reference"] = (
            "F_8/F_1_8 use the deterministic class-balanced real subset "
            f"with the first {sample_count // nclass} reference samples per class "
            f"({sample_count} total)"
        )
    if "KID" in headline_metrics:
        protocol["kid"] = {
            "feature_extractor": "pinned active-backbone InceptionV3 (2048-d)",
            "estimator": "unbiased cubic polynomial MMD",
            "subsets": args.kid_subsets,
            "max_subset_size": args.kid_subset_size,
            "subset_rng_seed": args.kid_seed,
        }
    headline_mode = "quick" if args.mode == "quick" else "headline"
    headline_payload = {
        "metrics": headline_metrics,
        "protocol": _headline_protocol(protocol, mode=headline_mode),
        "label_histogram": counts.tolist(),
    }
    # The primary path becomes available as a valid headline report immediately;
    # detailed mode atomically replaces it only after all optional work succeeds.
    _atomic_write_json(args.output, headline_payload)
    if args.headline_output is not None:
        _atomic_write_json(args.headline_output, headline_payload)
    if args.mode == "quick":
        if args.per_class_output is not None:
            _atomic_write_json(
                args.per_class_output,
                _per_class_payload(args, generated_features, reference_features, labels, nclass, dataset),
            )
        return headline_payload
    if args.mode == "headline":
        return headline_payload

    metrics = dict(headline_metrics)
    reference_vgg_features = np.load(args.metrics_root / f"{dataset}_vgg16_fc2.npy", mmap_mode="r")
    reference_radii = np.load(args.metrics_root / f"{dataset}_vgg16_fc2_k3_radii.npy", mmap_mode="r")
    precision, recall = improved_prd_vgg16_k3(
        np.asarray(images),
        np.asarray(reference_vgg_features),
        np.asarray(reference_radii),
        batch_size=args.vgg_batch_size,
        query_batch=args.knn_query_batch,
        device=getattr(args, "device", "cuda"),
    )
    metrics["ImprovedPrecision"] = float(precision)
    metrics["Recall"] = float(recall)
    if args.per_class_output is not None:
        _atomic_write_json(
            args.per_class_output,
            _per_class_payload(args, generated_features, reference_features, labels, nclass, dataset),
        )
    payload = {"metrics": metrics, "protocol": protocol, "label_histogram": counts.tolist()}
    _atomic_write_json(args.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-type", choices=("cifar10lt", "cifar100lt"), required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("detailed", "headline", "quick", "gpu_fast", "improved_pr"), default="detailed",
        help=("detailed preserves canonical 50k VGG/per-class behavior; headline emits "
              "canonical common metrics without details; quick accepts a smaller class-uniform "
              "smoke sample; gpu_fast emits a 50k CUDA preview without Inception PRD/VGG; "
              "improved_pr emits only canonical VGG improved Precision/Recall"),
    )
    parser.add_argument(
        "--headline-output", type=Path, default=None,
        help="optional permanent atomic headline JSON; --output is also written early then replaced by the detailed result",
    )
    parser.add_argument(
        "--inception-batch-size", type=int, default=16,
        help="GPU micro-batch for the pinned Inception FID/IS/PRD/KID feature pass (default: 16)",
    )
    parser.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda",
        help="device for Inception/VGG features and exact k-NN metrics; CPU mode avoids shared-GPU memory pressure",
    )
    parser.add_argument(
        "--generated-features-cache", type=Path, default=None,
        help="optional atomic .npz cache for generated 2048-d Inception features and IS",
    )
    parser.add_argument("--vgg-batch-size", type=int, default=128)
    parser.add_argument("--knn-query-batch", type=int, default=1024)
    parser.add_argument("--kid", action="store_true", help="append deterministic cubic-kernel KID in detailed mode (always included in headline and quick modes)")
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    parser.add_argument("--kid-seed", type=int, default=2026)
    parser.add_argument("--per-class-output", type=Path, default=None,
                        help="optional JSON with per-class FID and Many/Medium/Few FIDs (detailed and quick modes)")
    parser.add_argument("--longtail-groups", choices=("none", "cm_three_way"), default="none")
    args = parser.parse_args()
    if args.inception_batch_size <= 0:
        parser.error("--inception-batch-size must be positive")
    if args.vgg_batch_size <= 0:
        parser.error("--vgg-batch-size must be positive")
    if args.knn_query_batch <= 0:
        parser.error("--knn-query-batch must be positive")
    if args.kid_subsets <= 0:
        parser.error("--kid-subsets must be positive")
    if args.kid_subset_size <= 1:
        parser.error("--kid-subset-size must be at least two")
    if args.mode == "headline" and args.per_class_output is not None:
        parser.error("--mode headline intentionally omits per-class FID; rerun with --mode detailed for --per-class-output")
    if args.mode == "detailed" and args.headline_output is not None and args.headline_output == args.output:
        parser.error("--headline-output must differ from --output in detailed mode so the headline result remains available")

    args.samples = Path(args.samples).resolve()
    if args.generated_features_cache is not None:
        args.generated_features_cache = Path(args.generated_features_cache).resolve()
    images = np.load(args.samples, mmap_mode="r")
    labels = np.load(args.labels, mmap_mode="r")
    nclass = 100 if args.data_type == "cifar100lt" else 10
    if args.mode in ("detailed", "headline"):
        if images.shape != (50000, 3, 32, 32) or labels.shape != (50000,):
            raise ValueError(f"paper protocol requires 50k CIFAR arrays, got images={images.shape} labels={labels.shape}")
    elif args.mode == "improved_pr":
        if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
            raise ValueError(f"improved_pr requires images with shape [N, 3, 32, 32], got {images.shape}")
        if labels.ndim != 1 or labels.shape != (images.shape[0],) or images.shape[0] < 3:
            raise ValueError(f"improved_pr requires matching labels and at least 3 images, got {images.shape}/{labels.shape}")
    else:
        if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
            raise ValueError(f"quick protocol requires images with shape [N, 3, 32, 32], got {images.shape}")
        if labels.ndim != 1 or labels.shape != (images.shape[0],):
            raise ValueError(f"quick protocol requires labels with shape [N], got {labels.shape} for N={images.shape[0]}")
        if images.shape[0] <= 0 or images.shape[0] % nclass != 0:
            raise ValueError(
                f"quick protocol requires a positive sample count divisible by {nclass}, got N={images.shape[0]}"
            )
    labels_array = np.asarray(labels)
    counts = np.bincount(np.asarray(labels_array, dtype=np.int64), minlength=nclass)
    expected_count = images.shape[0] // nclass
    if len(counts) != nclass or np.any(counts != expected_count):
        raise ValueError(f"labels are not exactly class-uniform: {counts.tolist()}")
    os.environ["LTX_METRICS_ROOT"] = str(args.metrics_root.resolve())
    payload = _evaluate(args, images, labels_array, counts, nclass)
    print(json.dumps(payload["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
