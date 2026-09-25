"""Small wrappers around nvidia-smi and GPU slot calculation."""

from __future__ import annotations

import csv
import io
import math
import subprocess
from dataclasses import dataclass


@dataclass
class GPU:
    index: int
    name: str
    memory_free_mb: float


def query_gpus() -> list[GPU]:
    fields = "index,name,memory.free"
    proc = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if proc.returncode != 0:
        return []
    gpus: list[GPU] = []
    for row in csv.reader(io.StringIO(proc.stdout)):
        if len(row) < 3:
            continue
        try:
            gpus.append(GPU(
                index=int(row[0].strip()), name=row[1].strip(),
                memory_free_mb=float(row[2]),
            ))
        except ValueError:
            continue
    return gpus


def query_compute_apps() -> dict[int, float]:
    """Map pid -> used_gpu_memory_mb for every process nvidia-smi can see.

    Used to measure a single task's own footprint instead of the whole
    device, since packing means device-level memory is shared between
    several unrelated tasks.
    """
    proc = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if proc.returncode != 0:
        return {}
    apps: dict[int, float] = {}
    for row in csv.reader(io.StringIO(proc.stdout)):
        if len(row) < 2:
            continue
        try:
            apps[int(row[0].strip())] = float(row[1].strip())
        except ValueError:
            continue
    return apps


def plan_slots(
    gpus: list[GPU],
    *,
    tasks_per_gpu: "int | str" = "auto",
    task_memory_gb: float = 12.0,
    headroom_gb: float = 4.0,
    ceiling: int = 4,
    running_on: dict[int, int] | None = None,
) -> dict[int, int]:
    """Decide how many concurrent tasks each GPU may host.

    ``tasks_per_gpu`` an int pins every GPU to that count. ``"auto"`` divides
    each GPU's capacity (minus a safety headroom) by ``task_memory_gb`` and
    clamps the result to ``[1, ceiling]`` so a single GPU is never starved nor
    over-packed past a sane bound.

    Capacity adds back the memory our own already-running tasks occupy
    (``running_on``). Measuring raw free memory instead would shrink the
    baseline as we fill the card, so a GPU would stall one or two slots below
    its real capacity and never reach ``ceiling``. Memory used by *other*
    tenants is deliberately not added back — that is not ours to reclaim.
    """
    running_on = running_on or {}
    slots: dict[int, int] = {}
    for gpu in gpus:
        if isinstance(tasks_per_gpu, int):
            slots[gpu.index] = max(1, tasks_per_gpu)
            continue
        ours_gb = running_on.get(gpu.index, 0) * task_memory_gb
        usable_gb = (gpu.memory_free_mb / 1024.0) + ours_gb - headroom_gb
        estimated = math.floor(usable_gb / task_memory_gb) if task_memory_gb > 0 else 1
        slots[gpu.index] = max(1, min(ceiling, estimated))
    return slots
