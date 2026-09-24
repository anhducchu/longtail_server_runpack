"""Small, framework-free helpers for explicit checkpoint provenance.

Checkpoint discovery is deliberately separate from checkpoint loading.  A run
may discover a checkpoint in its own run directory for the upstream resume
path, but a checkpoint outside that directory is only ever accepted when the
caller names it explicitly in config or on the CLI.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping


# Arbitrarily renamed files require an explicit resume_step, so a path can
# never silently pick an unrelated update number.
_CHECKPOINT_NAME = re.compile(r"^ckpt_(\d+)\.pt$")


def get_resume_checkpoint(
    train: Mapping[str, Any], method_config: Mapping[str, Any] | None = None
) -> Path | None:
    """Resolve an explicit full-state checkpoint from task configuration.

    Method-level values are useful for a multi-method stage and take
    precedence over stage-level values. External warm starts are intentionally
    unsupported: every accepted checkpoint resumes model, EMA, optimizer and
    scheduler state together.
    """
    values: dict[str, Any] = dict(train)
    if method_config:
        for key in ("resume_checkpoint", "resume_step"):
            if key in method_config:
                values[key] = method_config[key]

    raw_path = str(values.get("resume_checkpoint", "") or "").strip()
    if not raw_path:
        return None

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"explicit resume checkpoint does not exist: {path}. "
            "Set the path for the intended task, or omit it to use only the "
            "task run directory's normal local resume behavior."
        )
    return path


def get_resume_step(
    train: Mapping[str, Any],
    method_config: Mapping[str, Any] | None,
    checkpoint: Path,
) -> int:
    """Resolve the completed-update number for an explicit checkpoint.

    The checkpoint filename is the normal source of truth.  A separate
    ``resume_step`` is supported for files that were renamed, but it
    must be supplied explicitly rather than guessed from file contents.
    """
    values: dict[str, Any] = dict(train)
    if method_config:
        for key in ("resume_step",):
            if key in method_config:
                values[key] = method_config[key]
    raw_step = str(values.get("resume_step", "") or "").strip()
    if raw_step:
        try:
            step = int(raw_step)
        except ValueError as exc:
            raise ValueError(
                f"resume_step must be an integer, got {raw_step!r} for {checkpoint}"
            ) from exc
    else:
        match = _CHECKPOINT_NAME.fullmatch(checkpoint.name)
        if not match:
            raise ValueError(
                f"cannot infer resume step from {checkpoint.name!r}; set "
                "resume_step explicitly in the method config or CLI"
            )
        step = int(match.group(1))
    if step < 0:
        raise ValueError(f"resume_step must be non-negative, got {step}")
    return step
