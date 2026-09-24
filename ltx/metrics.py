from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict


def _put(out: Dict[str, float], key: str, value: object) -> None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(scalar):
        out[key] = scalar


def _flatten(prefix: str, value: object, out: Dict[str, float]) -> None:
    if isinstance(value, dict):
        if "mean" in value and isinstance(value["mean"], (int, float)):
            _put(out, prefix, value["mean"])
            return
        for key, child in value.items():
            _flatten(f"{prefix}/{key}" if prefix else str(key), child, out)
    elif isinstance(value, (int, float)):
        _put(out, prefix, value)

# Sample counts, seeds and protocol constants are provenance, not results.
# They belong in the run's config and the report's JSON, not plotted next to
# FID as if they were something a reader should compare across methods.
_BOOKKEEPING = ("num_generated", "num_reference", "generated", "reference",
                "seed", "samples", "label_histogram")


def _is_bookkeeping(key: str) -> bool:
    return key.rsplit("/", 1)[-1] in _BOOKKEEPING


def collect_metrics(run_dir: Path) -> Dict[str, float]:
    """Collect only JSON artifacts emitted by the shared baseline evaluator."""
    out: Dict[str, float] = {}
    for path in run_dir.rglob("metrics*.json"):
        if path.name == "metrics.collected.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("metrics"), dict):
            _flatten("generation", payload["metrics"], out)
        elif isinstance(payload.get("groups"), dict):
            _flatten("generation/tail", payload["groups"], out)
    return {k: v for k, v in out.items() if not _is_bookkeeping(k)}


def latest_tensorboard_scalars(logdir: Path):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return {}
    if not list(logdir.rglob("events.out.tfevents.*")):
        return {}
    try:
        acc = EventAccumulator(str(logdir), size_guidance={"scalars": 1})
        acc.Reload()
        out = {}
        for tag in acc.Tags().get("scalars", []):
            events = acc.Scalars(tag)
            if events:
                out[tag] = (int(events[-1].step), float(events[-1].value))
        return out
    except Exception:
        return {}
