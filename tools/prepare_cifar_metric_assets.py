#!/usr/bin/env python3
"""Build balanced CIFAR reference assets for a pinned metric host.

This is deliberately an external preparation step, not part of any model run:
it extracts Inception features from all 50k images of the *balanced* CIFAR
training split, then writes the FID moments and source-PRD feature cache. The
result is shared across every seed/method and is fingerprinted in a JSON
sidecar. It needs one CUDA GPU and uses the active CCUA score implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from runner.metrics import knn_radii, vgg16_fc2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(repo: Path, data_root: Path, output: Path, batch_size: int) -> None:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import CIFAR100

    sys.path.insert(0, str(repo))
    from score.inception import InceptionV3

    dataset_name = "cifar100"
    dataset = CIFAR100(root=str(data_root), train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    device = torch.device("cuda")
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    features = np.empty((len(dataset), 2048), dtype=np.float32)
    offset = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            batch = model(images)[0].flatten(1).cpu().numpy().astype(np.float32, copy=False)
            features[offset:offset + len(batch)] = batch
            offset += len(batch)
    if offset != len(dataset):
        raise RuntimeError(f"feature extraction incomplete: {offset} != {len(dataset)}")

    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / f"{dataset_name}_feats.npy"
    labels_path = output / f"{dataset_name}_labels.npy"
    fid_path = output / f"{dataset_name}.train.npz"
    np.save(feature_path, features)
    # Preserve class membership next to the balanced Inception cache.  This
    # lets the common evaluator compute per-class and Many/Medium/Few FIDs
    # without ever rebuilding a reference from the imbalanced training split.
    np.save(labels_path, np.asarray(dataset.targets, dtype=np.int64))
    np.savez(fid_path, mu=features.mean(axis=0, dtype=np.float64), sigma=np.cov(features, rowvar=False))
    # Paper Recall uses VGG16 fc2 features and a k=3 manifold, not an
    # Inception approximation.
    raw_images = np.asarray(dataset.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
    vgg_features = vgg16_fc2(raw_images, batch_size=min(batch_size, 64))
    vgg_path = output / f"{dataset_name}_vgg16_fc2.npy"
    vgg_radii_path = output / f"{dataset_name}_vgg16_fc2_k3_radii.npy"
    np.save(vgg_path, vgg_features)
    np.save(vgg_radii_path, knn_radii(vgg_features, k=3))
    # The released runpack deliberately vendors source without .git metadata.
    # Preserve provenance from the checked-in vendor manifest instead of
    # silently requiring the original checkout. Native CCUA is also the metric
    # host, so the metric cache lives beside the active backbone.
    manifest_path = repo.parent / "THIRD_PARTY_MANIFEST.json"
    try:
        components = json.loads(manifest_path.read_text(encoding="utf-8"))["components"]
        if repo.name != "CCUA-DDPM":
            raise RuntimeError(f"active metric preparation requires CCUA-DDPM, got {repo}")
        component = components["ccua"]
        commit = component["commit"]
    except Exception as exc:
        raise RuntimeError(f"vendored CCUA provenance missing: {manifest_path}: {exc}") from exc
    manifest = {
        "dataset": dataset_name,
        "split": "balanced CIFAR training (50,000 images, class-uniform source)",
        "num_images": int(len(dataset)),
        "feature_extractor": f"pinned {repo.name} InceptionV3 (2048-d)",
        "metric_protocol": "shared FID and PRD feature cache",
        "repository": str(repo),
        "repository_commit": commit,
        "feature_file": feature_path.name,
        "feature_sha256": sha256(feature_path),
        "labels_file": labels_path.name,
        "labels_sha256": sha256(labels_path),
        "fid_file": fid_path.name,
        "fid_sha256": sha256(fid_path),
        "improved_prd_feature_extractor": "torchvision VGG16 ImageNet fc2 (4096-d), source-compatible resize to 224",
        "improved_prd_k": 3,
        "vgg_feature_file": vgg_path.name,
        "vgg_feature_sha256": sha256(vgg_path),
        "vgg_radii_file": vgg_radii_path.name,
        "vgg_radii_sha256": sha256(vgg_radii_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / f"{dataset_name}.metric_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if not args.repo.joinpath("score", "inception.py").is_file():
        raise FileNotFoundError(f"metric host is missing score/inception.py: {args.repo}")
    required = (
        args.output / "cifar100.train.npz",
        args.output / "cifar100_feats.npy",
        args.output / "cifar100_labels.npy",
        args.output / "cifar100_vgg16_fc2.npy",
        args.output / "cifar100_vgg16_fc2_k3_radii.npy",
        args.output / "cifar100.metric_manifest.json",
    )
    if all(path.exists() for path in required):
        print(f"[metric-assets] keeping existing CIFAR-100 assets in {args.output}")
    else:
        extract(args.repo.resolve(), args.data_root.resolve(), args.output.resolve(), args.batch_size)


if __name__ == "__main__":
    main()
