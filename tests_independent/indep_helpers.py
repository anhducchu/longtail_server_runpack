"""Oracles written from the maths in scaffold/README.md, not from the scaffold code.

Everything here is float64 numpy. Nothing imports ``scaffold``.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

CLEAN = -1
MARGIN = 1e-4


def np_alpha_bar(steps: int = 1000, beta_1: float = 1e-4, beta_T: float = 0.02) -> np.ndarray:
    return np.cumprod(1.0 - np.linspace(beta_1, beta_T, steps, dtype=np.float64))


def ab_at(alpha_bar: np.ndarray, index: Any) -> np.ndarray:
    """alpha_bar per index, CLEAN -> 1, shaped [B, 1, 1, 1] for arrays."""
    index = np.asarray(index)
    values = np.where(index < 0, 1.0, alpha_bar[np.maximum(index, 0)])
    return values.reshape(-1, 1, 1, 1) if index.ndim else values


def np_beta_tilde(ab_t: Any, ab_s: Any) -> Any:
    return (1.0 - ab_s) / (1.0 - ab_t) * (1.0 - ab_t / ab_s)


def np_edge(x_t: np.ndarray, eps: np.ndarray, lam: Any, ab_t: Any, ab_s: Any):
    """README edge: returns (x0_hat, mean, variance, capped)."""
    x0_hat = (x_t - np.sqrt(1.0 - ab_t) * eps) / np.sqrt(ab_t)
    requested = lam * np_beta_tilde(ab_t, ab_s)
    budget = (1.0 - MARGIN) * (1.0 - ab_s)
    variance = np.minimum(requested, budget)
    mean = np.sqrt(ab_s) * x0_hat + np.sqrt(1.0 - ab_s - variance) * eps
    return x0_hat, mean, variance, requested > budget


def np_posterior_bayes(x0: np.ndarray, x_t: np.ndarray, ab_t: Any, ab_s: Any):
    """q(x_s | x_t, x0) from Bayes' rule: q(x_t | x_s) q(x_s | x0), product of Gaussians in x_s."""
    ratio = ab_t / ab_s
    precision = ratio / (1.0 - ratio) + 1.0 / (1.0 - ab_s)
    variance = 1.0 / precision
    mean = variance * (np.sqrt(ratio) * x_t / (1.0 - ratio) + np.sqrt(ab_s) * x0 / (1.0 - ab_s))
    return mean, variance


def np_gaussian_kl(mean_q, var_q, mean_p, var_p) -> np.ndarray:
    """Per-coordinate KL(N(mean_q, var_q) || N(mean_p, var_p))."""
    return 0.5 * (np.log(var_p / var_q) + (var_q + (mean_q - mean_p) ** 2) / var_p - 1.0)


def protocol_generator(seed: int, stream: int, batch: int) -> torch.Generator:
    """'SeedSequence(seed, stream, batch)' as stated in PROTOCOL.json / the README."""
    state = np.random.SeedSequence([seed, stream, batch]).generate_state(1, np.uint64)[0]
    return torch.Generator().manual_seed(int(state) & (2 ** 63 - 1))


# Tiny CPU problem shared by the end-to-end tests

TINY_STEPS, TINY_B1, TINY_BT = 50, 1e-4, 0.2
TINY_AB = np_alpha_bar(TINY_STEPS, TINY_B1, TINY_BT)


def np_fake_epsilon(x: np.ndarray, t: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """A wrong-but-smooth 'score': exact for N(0, I) data, plus a label-dependent shift."""
    scale = np.sqrt(1.0 - TINY_AB[t]).reshape(-1, 1, 1, 1)
    return x * scale + 0.05 * labels.reshape(-1, 1, 1, 1)


class FakeEpsilon:
    """Torch twin of ``np_fake_epsilon`` that records the timesteps it was asked for."""

    def __init__(self) -> None:
        self.alpha_bar = torch.from_numpy(TINY_AB)
        self.timesteps: list[int] = []

    def __call__(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.timesteps.append(int(t[0]))
        assert int(t.min()) >= 0, 'theta must never be evaluated at CLEAN'
        scale = (1.0 - self.alpha_bar[t.cpu()]).sqrt().to(x).reshape(-1, 1, 1, 1)
        return x * scale + 0.05 * labels.to(x).reshape(-1, 1, 1, 1)


def tiny_config() -> dict[str, Any]:
    return copy.deepcopy({
        'paths': {'checkpoint': 'unused.pt', 'checkpoint_sha256': 'unused'},
        'dataset': {'root': 'unused', 'imbalance_factor': 0.01, 'split_seed': 0, 'num_classes': 4, 'image_size': 8},
        'diffusion': {'steps': TINY_STEPS, 'beta_1': TINY_B1, 'beta_T': TINY_BT},
        'model': {'microbatch': 16},
        'sampler': {'grid': {'nfe': 5, 'spacing': 'leading'}, 'guidance_scale': 0.0},
        'scaffold': {'kind': 'mlp', 'inputs': 'state_time_class', 'image_size': 8, 'hidden_dims': [16],
                     'condition_dim': 8, 'variance_floor': 1e-4, 'variance_initial': 1.0},
        'loss': {'kind': 'respaced_vlb'},
        'training': {'name': 'tiny_train', 'seed': 3, 'classes_per_batch': 2, 'samples_per_class': 4,
                     'updates': 6, 'learning_rate': 1e-2, 'weight_decay': 0.0, 'gradient_clip': 1.0,
                     'horizontal_flip': True, 'ema_decay': 0.9, 'checkpoint_interval': 3, 'log_interval': 1000,
                     'retain_snapshots': True},
        'validation': {'per_class': 3, 'seed': 5, 'batch_size': 7, 'heldout': False},
        'evaluation': {'name': 'tiny_eval', 'num_images': 8, 'batch_size': 3, 'seeds': [0, 1],
                       'metrics_root': 'unused', 'inception_batch_size': 4, 'vgg_batch_size': 4, 'recall': False,
                       'arms': [{'name': 'ddim', 'kind': 'fixed', 'lambda': 0.0},
                                {'name': 'half', 'kind': 'fixed', 'lambda': 0.5},
                                {'name': 'eta1', 'kind': 'fixed', 'lambda': 1.0},
                                {'name': 'small', 'kind': 'ddpm', 'variance_type': 'fixedsmall'},
                                {'name': 'large', 'kind': 'ddpm', 'variance_type': 'fixedlarge'}]},
    })


def tiny_dataset(seed: int = 0):
    """(images uint8 NHWC, per-class index lists, counts) with a long tail over 4 classes."""
    rng = np.random.default_rng(seed)
    counts = np.array([12, 8, 5, 2])
    labels = np.repeat(np.arange(4), counts)
    # Smooth, low-variance images so that the N(0, I) fake score is clearly wrong.
    base = rng.integers(60, 200, size=(len(labels), 1, 1, 3))
    images = np.clip(base + rng.integers(-20, 20, size=(len(labels), 8, 8, 3)), 0, 255).astype(np.uint8)
    indices = [np.flatnonzero(labels == c) for c in range(4)]
    return images, indices, counts.astype(np.float32)
