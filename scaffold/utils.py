"""Paths, configuration, atomic I/O, hashing, seeding and the CIFAR-100-LT data."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torchvision.datasets as datasets

ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = SCAFFOLD_ROOT / 'outputs'
CONFIG_SECTIONS = ('paths', 'dataset', 'diffusion', 'model', 'sampler', 'scaffold', 'loss',
                   'training', 'validation', 'evaluation')
_SAFE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')


# Configuration and paths

def load_config(path: Path, overrides: tuple[str, ...] | list[str] = ()) -> dict[str, Any]:
    """Read a JSON config and apply ``section.key=value`` overrides.

    Override values are parsed as JSON when possible (``sampler.grid.nfe=50``,
    ``sampler.grid={"stride": 10}``) and kept as strings otherwise.
    """
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    for override in overrides:
        key, separator, raw = override.partition('=')
        if not separator:
            raise ValueError(f'override must look like section.key=value, got {override!r}')
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        *parents, leaf = key.split('.')
        node = config
        for name in parents:
            node = node[name]
        node[leaf] = value
    missing = [name for name in CONFIG_SECTIONS if name not in config]
    if missing:
        raise ValueError(f'config is missing sections: {missing}')
    return config


def rooted_path(value: str | Path) -> Path:
    """Resolve a config path; relative paths are relative to the repository root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def output_directory(name: str) -> Path:
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError("run names may contain only letters, numbers, '.', '_' and '-'")
    return OUTPUTS_ROOT / name


# Hashing

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """SHA-256 of a JSON-serialisable value, independent of key order."""
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def source_hashes() -> dict[str, str]:
    """SHA-256 of every source and config file of this package, recorded with each result."""
    files = sorted(SCAFFOLD_ROOT.glob('*.py')) + [SCAFFOLD_ROOT / 'config.json']
    return {path.name: sha256_file(path) for path in files}


def array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode('utf-8'))
    digest.update(str(values.shape).encode('utf-8'))
    digest.update(values.tobytes())
    return digest.hexdigest()


# Atomic output: a file appears under its final name only once fully written.

@contextmanager
def atomic_output(path: Path, *, suffix: str = '') -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix=suffix, dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Any) -> None:
    with atomic_output(path) as temporary:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')


def atomic_numpy(path: Path, array: np.ndarray) -> None:
    with atomic_output(path, suffix='.npy') as temporary:
        with temporary.open('wb') as stream:
            np.save(stream, array, allow_pickle=False)


def atomic_npz(path: Path, **arrays: Any) -> None:
    with atomic_output(path, suffix='.npz') as temporary:
        with temporary.open('wb') as stream:
            np.savez(stream, **arrays)


def atomic_torch_save(path: Path, payload: Any) -> None:
    with atomic_output(path, suffix='.pt') as temporary:
        torch.save(payload, temporary)


# Randomness

def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stream_generator(device: torch.device, *key: int) -> torch.Generator:
    """Independent generator for one key, e.g. (seed, stream, batch).

    Keys go through numpy's SeedSequence, so nearby keys give unrelated streams;
    seed and seed+1 never share a batch.
    """
    state = np.random.SeedSequence([int(part) for part in key]).generate_state(1, np.uint64)[0]
    return torch.Generator(device=device).manual_seed(int(state) & (2 ** 63 - 1))


# Data

class ImbalanceCIFAR100(datasets.CIFAR100):
    """Long-tailed CIFAR-100 subset, identical to the split the frozen score was trained on."""

    cls_num = 100

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        self.num_per_cls_dict: dict[int, int] = {}
        self._make_imbalanced(self._image_counts(imb_type, imb_factor))

    def _image_counts(self, imbalance_type: str, imbalance_factor: float) -> list[int]:
        maximum = len(self.data) / self.cls_num
        if imbalance_type == 'exp':
            return [int(maximum * imbalance_factor ** (index / (self.cls_num - 1.0)))
                    for index in range(self.cls_num)]
        if imbalance_type == 'step':
            half = self.cls_num // 2
            return [int(maximum)] * half + [int(maximum * imbalance_factor)] * half
        return [int(maximum)] * self.cls_num

    def _make_imbalanced(self, image_counts: list[int]) -> None:
        targets = np.asarray(self.targets, dtype=np.int64)
        data, labels = [], []
        for class_id, image_count in zip(np.unique(targets), image_counts):
            self.num_per_cls_dict[int(class_id)] = image_count
            indices = np.flatnonzero(targets == class_id)
            np.random.shuffle(indices)
            data.append(self.data[indices[:image_count], ...])
            labels.extend([int(class_id)] * image_count)
        self.data = np.vstack(data)
        self.targets = labels


def load_training_data(config: dict[str, Any]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Long-tailed training images (uint8 NHWC), per-class index lists and class counts."""
    settings = config['dataset']
    dataset = ImbalanceCIFAR100(root=str(rooted_path(settings['root'])), imb_type='exp',
                                imb_factor=float(settings['imbalance_factor']),
                                rand_number=int(settings['split_seed']), train=True, download=False)
    images = np.asarray(dataset.data, dtype=np.uint8)
    labels = np.asarray(dataset.targets, dtype=np.int64)
    classes = int(settings['num_classes'])
    indices = [np.flatnonzero(labels == class_id) for class_id in range(classes)]
    if any(len(class_indices) == 0 for class_indices in indices):
        raise ValueError('the long-tailed training split contains an empty class')
    return images, indices, np.bincount(labels, minlength=classes).astype(np.float32)


def load_heldout_data(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Balanced CIFAR-100 test images and labels. Diagnostics only; never trained on."""
    dataset = datasets.CIFAR100(root=str(rooted_path(config['dataset']['root'])), train=False, download=False)
    return np.asarray(dataset.data, dtype=np.uint8), np.asarray(dataset.targets, dtype=np.int64)


def to_pixels(images: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 NHWC -> float32 NCHW in [-1, 1]."""
    tensor = torch.from_numpy(np.ascontiguousarray(images)).to(device=device, dtype=torch.float32)
    return tensor.permute(0, 3, 1, 2).div(127.5).sub(1.0)
