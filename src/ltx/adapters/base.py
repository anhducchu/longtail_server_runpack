from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from ..config import Task


def resolve_num_workers(train: Dict, default: int) -> int:
    """Dataloader worker count, letting the scheduler's packing-aware
    LTX_NUM_WORKERS env override the campaign config so co-located tasks
    don't oversubscribe the host's CPUs."""
    override = os.environ.get("LTX_NUM_WORKERS")
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    return int(train.get("num_workers", default))


def resolve_inception_batch_size(evaluate: Dict, default: int = 16) -> int:
    """Validate the GPU micro-batch used by the shared Inception evaluator.

    This is independent of the training and sampling batch sizes.  Keeping it
    small by default makes the final metric phase robust when several workers
    share a GPU, without changing the number or identity of metric samples.
    """
    value = int(evaluate.get("inception_batch_size", default))
    if value <= 0:
        raise ValueError(f"inception_batch_size must be positive, got {value}")
    return value


@dataclass
class Phase:
    name: str
    command: List[str]
    cwd: Path
    env: Dict[str, str] = field(default_factory=dict)
    skip_if_exists: List[Path] = field(default_factory=list)


class Adapter:
    def __init__(self, root: Path):
        self.root = root

    def phases(self, task: Task, batch_size: int | None = None) -> List[Phase]:
        raise NotImplementedError

    def repo_dir(self, task: Task) -> Path:
        directory = task.repository["directory"]
        path = Path(directory)
        return path if path.is_absolute() else Path(task.runtime["repos_root"]) / path
