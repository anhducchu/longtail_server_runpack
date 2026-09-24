"""Shared paper-facing generation metrics.

This module implements the paper evaluator shared by all four objectives. It
uses chunked exact squared distances, avoiding a
50k×50k host allocation.

``polynomial_mmd_kid`` is an unbiased cubic-kernel estimate. This runpack
accepts an explicit generator so a KID value is reproducible across jobs.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np


def polynomial_mmd_kid(
    generated_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    num_subsets: int = 100,
    max_subset_size: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """Compute KID with the cubic polynomial kernel estimator.

    This is algebraically identical to the released reference implementation
    except that subset draws use a caller-controlled random generator.  It is
    intentionally a *single* KID estimate, not an artificial standard
    deviation based on repeatedly re-evaluating one trained model.
    """
    generated = np.asarray(generated_features, dtype=np.float32)
    reference = np.asarray(reference_features, dtype=np.float32)
    if generated.ndim != 2 or reference.ndim != 2 or generated.shape[1] != reference.shape[1]:
        raise ValueError(
            "KID expects generated/reference arrays with matching [N, feature_dim] shapes, got "
            f"{generated.shape} and {reference.shape}"
        )
    subset = min(len(generated), len(reference), int(max_subset_size))
    if subset < 2:
        raise ValueError("KID requires at least two generated and two reference feature vectors")
    if int(num_subsets) < 1:
        raise ValueError("KID requires num_subsets >= 1")
    generator = rng or np.random.default_rng(0)
    dim = generated.shape[1]
    total = 0.0
    for _ in range(int(num_subsets)):
        x = generated[generator.choice(len(generated), subset, replace=False)]
        y = reference[generator.choice(len(reference), subset, replace=False)]
        a = (x @ x.T / dim + 1.0) ** 3 + (y @ y.T / dim + 1.0) ** 3
        b = (x @ y.T / dim + 1.0) ** 3
        total += (a.sum() - np.diag(a).sum()) / (subset - 1) - 2.0 * b.sum() / subset
    return float(total / int(num_subsets) / subset)


def polynomial_mmd_kid_gpu(
    generated_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    num_subsets: int = 100,
    max_subset_size: int = 1000,
    rng: np.random.Generator | None = None,
    device=None,
) -> float:
    """Compute the same cubic KID estimator with CUDA matrix multiplies.

    NumPy chooses the subset indices exactly as in :func:`polynomial_mmd_kid`;
    only the Gram-matrix arithmetic is moved to torch/CUDA.  This is intended
    for fast monitoring.  The canonical paper path remains the NumPy
    implementation so its historical numbers are not silently changed by a
    backend switch.
    """
    generated = np.asarray(generated_features, dtype=np.float32)
    reference = np.asarray(reference_features, dtype=np.float32)
    if generated.ndim != 2 or reference.ndim != 2 or generated.shape[1] != reference.shape[1]:
        raise ValueError(
            "KID expects generated/reference arrays with matching [N, feature_dim] shapes, got "
            f"{generated.shape} and {reference.shape}"
        )
    subset = min(len(generated), len(reference), int(max_subset_size))
    if subset < 2:
        raise ValueError("KID requires at least two generated and two reference feature vectors")
    if int(num_subsets) < 1:
        raise ValueError("KID requires num_subsets >= 1")

    torch = _torch()
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if getattr(device, "type", None) != "cuda":
        raise ValueError(f"GPU KID requires a CUDA device, got {device}")

    generator = rng or np.random.default_rng(0)
    dim = generated.shape[1]
    generated_gpu = torch.from_numpy(np.ascontiguousarray(generated)).to(device)
    reference_gpu = torch.from_numpy(np.ascontiguousarray(reference)).to(device)
    total = torch.zeros((), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for _ in range(int(num_subsets)):
            generated_indices = torch.from_numpy(
                np.asarray(generator.choice(len(generated), subset, replace=False), dtype=np.int64)
            ).to(device)
            reference_indices = torch.from_numpy(
                np.asarray(generator.choice(len(reference), subset, replace=False), dtype=np.int64)
            ).to(device)
            x = generated_gpu.index_select(0, generated_indices)
            y = reference_gpu.index_select(0, reference_indices)
            kxx = (x @ x.T / dim + 1.0).pow(3)
            kyy = (y @ y.T / dim + 1.0).pow(3)
            kxy = (x @ y.T / dim + 1.0).pow(3)
            total = total + (
                (kxx.sum() - torch.diagonal(kxx).sum()
                 + kyy.sum() - torch.diagonal(kyy).sum()) / (subset - 1)
                - 2.0 * kxy.sum() / subset
            )
    return float((total / int(num_subsets) / subset).item())


def _torch(device="cuda"):
    import torch
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("paper metrics require a CUDA GPU")
    return torch


def vgg16_fc2(images: np.ndarray, batch_size: int = 128, device="cuda") -> np.ndarray:
    """Match the released IPR extractor: resize to 224 then VGG16 fc2."""
    torch = _torch(device)
    target_device = torch.device(device)
    from torchvision import models
    try:
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
    except AttributeError:  # torchvision < 0.13
        model = models.vgg16(pretrained=True)
    model = model.to(target_device).eval()
    result = np.empty((len(images), 4096), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(images[start:start + batch_size])).float().to(
                target_device, non_blocking=target_device.type == "cuda"
            )
            if batch.shape[-2:] != (224, 224):
                batch = torch.nn.functional.interpolate(batch, size=(224, 224))
            features = model.features(batch).flatten(1)
            result[start:start + len(batch)] = model.classifier[:4](features).cpu().numpy()
    return result


def _dist2(query, bank, bank_norm):
    torch = _torch(query.device)
    out = query.square().sum(dim=1, keepdim=True) + bank_norm.unsqueeze(0) - 2 * (query @ bank.T)
    return out.clamp_min_(0)


def _faiss_gpu_module():
    """Return a GPU-enabled FAISS module when the optional dependency exists."""
    try:
        import faiss
    except ImportError:
        return None
    if not all(hasattr(faiss, name) for name in ("StandardGpuResources", "index_cpu_to_gpu")):
        return None
    return faiss


def _knn_radii_torch(
    features: np.ndarray,
    k: int,
    query_batch: int,
    *,
    device,
) -> np.ndarray:
    """Exact tiled k-NN radii using PyTorch GEMMs on ``device``."""
    import torch

    bank = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(device)
    bank_norm = bank.square().sum(dim=1)
    radii = np.empty(len(features), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(features), query_batch):
            query = bank[start:start + query_batch]
            distances = _dist2(query, bank, bank_norm)
            rows = torch.arange(len(query), device=bank.device)
            distances[rows, start + rows] = float("inf")
            radii[start:start + len(query)] = (
                torch.topk(distances, k=k, largest=False).values[:, -1].sqrt().cpu().numpy()
            )
    return radii


def _knn_radii_faiss(features: np.ndarray, k: int, *, torch, faiss) -> np.ndarray:
    """Exact self-k-NN radii through FAISS ``IndexFlatL2`` on the GPU.

    FAISS is optional.  The index is flat (not IVF/PQ), so this changes the
    implementation backend but not the nearest-neighbour definition.
    """
    values = np.ascontiguousarray(features, dtype=np.float32)
    resources = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatL2(values.shape[1])
    index = faiss.index_cpu_to_gpu(resources, torch.cuda.current_device(), cpu_index)
    try:
        index.add(values)
        distances, neighbours = index.search(values, min(len(values), k + 1))
        row_ids = np.arange(len(values), dtype=np.int64)[:, None]
        # Exclude the query's own vector explicitly.  The extra neighbour
        # keeps this correct for the usual case where self is returned first;
        # masking also handles tied/duplicate vectors without relying on tie
        # ordering.
        filtered = np.where(neighbours != row_ids, distances, np.inf)
        kth = np.partition(filtered, k - 1, axis=1)[:, k - 1]
        if not np.all(np.isfinite(kth)):
            raise RuntimeError("FAISS did not return enough non-self neighbours")
        return np.sqrt(np.maximum(kth, 0.0)).astype(np.float32, copy=False)
    finally:
        del index, cpu_index, resources


def knn_radii(features: np.ndarray, k: int = 3, query_batch: int = 1024, device="cuda") -> np.ndarray:
    """Exact k-NN manifold radii; the self-distance is explicitly excluded.

    ``LTX_METRICS_KNN_BACKEND=faiss`` forces the optional exact FAISS backend;
    ``torch`` forces the tiled PyTorch implementation.  ``auto`` (default)
    uses FAISS when a GPU build is installed and otherwise uses PyTorch.
    """
    if len(features) <= k:
        raise ValueError(f"k-NN radii require more than k samples, got n={len(features)} k={k}")
    torch = _torch(device)
    target_device = torch.device(device)
    if target_device.type == "cuda" and target_device.index is None:
        target_device = torch.device("cuda", torch.cuda.current_device())
    backend = os.environ.get("LTX_METRICS_KNN_BACKEND", "auto").lower()
    if backend not in {"auto", "faiss", "torch"}:
        raise ValueError("LTX_METRICS_KNN_BACKEND must be one of: auto, faiss, torch")
    if target_device.type == "cuda" and backend != "torch":
        faiss = _faiss_gpu_module()
        if faiss is None and backend == "faiss":
            raise RuntimeError("LTX_METRICS_KNN_BACKEND=faiss requested but GPU FAISS is unavailable")
        if faiss is not None:
            try:
                return _knn_radii_faiss(features, k, torch=torch, faiss=faiss)
            except (RuntimeError, ValueError) as exc:
                if backend == "faiss":
                    raise
                warnings.warn(
                    f"GPU FAISS k-NN failed ({exc}); falling back to exact PyTorch tiles",
                    RuntimeWarning,
                    stacklevel=2,
                )
    return _knn_radii_torch(features, k, query_batch, device=target_device)


def _fused_cross_membership_torch(
    generated_features: np.ndarray,
    reference_features: np.ndarray,
    generated_radii: np.ndarray,
    reference_radii: np.ndarray,
    *,
    query_batch: int,
    device,
) -> tuple[float, float]:
    """Compute exact precision and recall from one cross-distance pass.

    The old implementation formed the same generated/reference distance
    matrix twice, once in each direction.  A tiled matrix is symmetric, so a
    single GEMM can update both membership directions: generated points are
    tested against reference balls, while reference points are tested against
    generated balls.  The result is mathematically identical in float32.
    """
    import torch

    generated = torch.from_numpy(np.ascontiguousarray(generated_features, dtype=np.float32)).to(device)
    reference = torch.from_numpy(np.ascontiguousarray(reference_features, dtype=np.float32)).to(device)
    generated_norm = generated.square().sum(dim=1)
    reference_norm = reference.square().sum(dim=1)
    generated_radii2 = torch.from_numpy(np.ascontiguousarray(generated_radii, dtype=np.float32)).to(device).square()
    reference_radii2 = torch.from_numpy(np.ascontiguousarray(reference_radii, dtype=np.float32)).to(device).square()
    precision_hits = torch.zeros(len(generated), dtype=torch.bool, device=device)
    recall_hits = torch.zeros(len(reference), dtype=torch.bool, device=device)

    with torch.inference_mode():
        for start in range(0, len(generated), query_batch):
            end = min(start + query_batch, len(generated))
            distances = (
                generated[start:end].square().sum(dim=1, keepdim=True)
                + reference_norm.unsqueeze(0)
                - 2 * (generated[start:end] @ reference.T)
            ).clamp_min_(0)
            precision_hits[start:end] = (distances <= reference_radii2.unsqueeze(0)).any(dim=1)
            recall_hits |= (distances <= generated_radii2[start:end].unsqueeze(1)).any(dim=0)

    return float(precision_hits.float().mean().item()), float(recall_hits.float().mean().item())


def improved_prd_vgg16_k3(generated_images: np.ndarray, reference_features: np.ndarray,
                           reference_radii: np.ndarray, batch_size: int = 128, query_batch: int = 1024,
                           device="cuda") -> tuple[float, float]:
    torch = _torch(device)
    target_device = torch.device(device)
    if target_device.type == "cuda" and target_device.index is None:
        target_device = torch.device("cuda", torch.cuda.current_device())
    generated_features = vgg16_fc2(generated_images, batch_size=batch_size, device=target_device)
    generated_radii = knn_radii(
        generated_features, k=3, query_batch=query_batch, device=target_device
    )
    return _fused_cross_membership_torch(
        generated_features,
        reference_features,
        generated_radii,
        reference_radii,
        query_batch=query_batch,
        device=target_device,
    )
