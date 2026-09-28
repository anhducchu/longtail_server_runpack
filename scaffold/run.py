#!/usr/bin/env python3
"""Entrypoint for the isolated frozen-score scaffold experiment.

Flow: load EMA score -> freeze theta -> train phi -> paired sample -> evaluate.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from scaffold.common import (
    ROOT,
    SCAFFOLD_ROOT,
    atomic_json,
    load_config,
    load_frozen_score,
    output_directory,
    rooted_path,
    set_reproducible_seed,
    sha256_file,
)
from scaffold.sample import (
    evaluate_conditions,
    sample_conditions,
    sample_fixed_sweep,
    sweep_conditions,
    write_fixed_sweep_table,
)
from scaffold.train import train_scaffold


def run_fixed_parallel(
    args: argparse.Namespace,
    config_path: Path,
    score_checkpoint: Path,
    output: Path,
    config: dict,
) -> None:
    """Sample fixed conditions with a bounded worker queue, then evaluate once."""
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending = list(sweep_conditions(config))
    active = []
    completed = []
    worker_pids = {}
    failures = {}

    def start_worker(condition: str):
        command = [
            sys.executable,
            "-m",
            "scaffold.run",
            "fixed-condition",
            "--config",
            str(config_path),
            "--checkpoint",
            str(score_checkpoint),
            "--name",
            output.name,
            "--device",
            args.device,
            "--condition",
            condition,
        ]
        stream = (logs / f"{condition}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        worker_pids[condition] = process.pid
        atomic_json(output / "parallel_workers.json", worker_pids)
        return condition, process, stream

    while pending or active:
        while pending and len(active) < args.parallel_workers:
            condition = pending.pop(0)["name"]
            active.append(start_worker(condition))
        condition, process, stream = active.pop(0)
        code = process.wait()
        stream.close()
        completed.append((condition, code))
        if code:
            failures[condition] = code
        atomic_json(output / "parallel_exit_codes.json", dict(completed))
    if failures:
        raise RuntimeError(f"fixed-condition workers failed: {failures}")

    command = [
        sys.executable,
        "-m",
        "scaffold.run",
        "fixed-evaluate",
        "--config",
        str(config_path),
        "--checkpoint",
        str(score_checkpoint),
        "--name",
        output.name,
        "--parallel-workers",
        str(args.parallel_workers),
    ]
    with (logs / "evaluate.log").open("w", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_fixed_evaluate_parallel(
    args: argparse.Namespace,
    config_path: Path,
    score_checkpoint: Path,
    output: Path,
    config: dict,
) -> Path:
    """Evaluate fixed conditions with the same bounded worker queue."""
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending = list(sweep_conditions(config))
    active = []
    completed = []
    worker_pids = {}
    failures = {}

    def start_worker(condition: str):
        command = [
            sys.executable,
            "-m",
            "scaffold.run",
            "fixed-evaluate-condition",
            "--config",
            str(config_path),
            "--checkpoint",
            str(score_checkpoint),
            "--name",
            output.name,
            "--condition",
            condition,
        ]
        stream = (logs / f"{condition}.evaluate.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        worker_pids[condition] = process.pid
        atomic_json(output / "evaluate_workers.json", worker_pids)
        return condition, process, stream

    while pending or active:
        while pending and len(active) < args.parallel_workers:
            condition = pending.pop(0)["name"]
            active.append(start_worker(condition))
        condition, process, stream = active.pop(0)
        code = process.wait()
        stream.close()
        completed.append((condition, code))
        if code:
            failures[condition] = code
        atomic_json(output / "evaluate_exit_codes.json", dict(completed))
    if failures:
        raise RuntimeError(f"fixed-evaluate workers failed: {failures}")
    return write_fixed_sweep_table(output, config)


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = load_config(config_path)
    score_checkpoint = rooted_path(args.checkpoint)
    if not score_checkpoint.is_file():
        raise FileNotFoundError(score_checkpoint)
    output = output_directory(config, args.name)
    score_sha256 = sha256_file(score_checkpoint)

    if args.command == "plan":
        show_plan(config, score_checkpoint, output)
        return

    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "config.resolved.json",
        {
            "config": config,
            "score_checkpoint": str(score_checkpoint),
            "score_sha256": score_sha256,
            "output": str(output),
        },
    )
    if args.command == "evaluate":
        evaluate_conditions(config, output)
        return
    if args.command == "fixed-evaluate-condition":
        sweep_conditions(config, args.condition)
        evaluate_conditions(
            config,
            output,
            (args.condition,),
            mode="fixed_table",
            cache_features=False,
        )
        return
    if args.command == "fixed-evaluate":
        result = run_fixed_evaluate_parallel(
            args, config_path, score_checkpoint, output, config
        )
        print(f"[result] {result}")
        return
    if args.command == "fixed-parallel":
        run_fixed_parallel(args, config_path, score_checkpoint, output, config)
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("scaffold train/sample requires a CUDA GPU")
    set_reproducible_seed(int(config["experiment"]["seed"]))
    score_model = load_frozen_score(score_checkpoint, config, device)

    if args.command in {"train", "all"}:
        train_scaffold(
            score_model,
            score_checkpoint,
            score_sha256,
            config,
            output,
            device,
        )
    if args.command in {"sample", "all"}:
        sample_conditions(
            score_model,
            score_checkpoint,
            score_sha256,
            config,
            output,
            device,
        )
    if args.command == "fixed-sweep":
        sample_fixed_sweep(
            score_model,
            score_checkpoint,
            score_sha256,
            config,
            output,
            device,
        )
        fixed_conditions = tuple(row["name"] for row in sweep_conditions(config))
        evaluate_conditions(
            config,
            output,
            fixed_conditions,
            mode="fixed_table",
            cache_features=False,
        )
        result = write_fixed_sweep_table(output, config)
        print(f"[result] {result}")
    if args.command == "fixed-condition":
        sample_fixed_sweep(
            score_model,
            score_checkpoint,
            score_sha256,
            config,
            output,
            device,
            condition=args.condition,
        )
    if args.command == "all" and bool(config["evaluation"].get("enabled", False)):
        evaluate_conditions(config, output)


def show_plan(config: dict, checkpoint: Path, output: Path) -> None:
    condition_count = (
        len(sweep_conditions(config))
        if config["experiment"].get("kind") == "fixed_sweep"
        else 2
    )
    print(f"score checkpoint: {checkpoint}")
    print(f"frozen weights:   {config['model']['checkpoint_key']}")
    print(f"output:           {output}")
    print(f"updates:          {config['training']['updates']}")
    print(f"CFG omega:        {config['sampling']['guidance_scale']}")
    print("sample range:     [0, 1]")
    print(
        f"samples:          {config['sampling']['num_images']} "
        f"x {condition_count} paired conditions"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated stage-2 scaffold on a frozen EMA score checkpoint."
    )
    parser.add_argument(
        "command",
        choices=(
            "plan",
            "train",
            "sample",
            "evaluate",
            "all",
            "fixed-sweep",
            "fixed-condition",
            "fixed-evaluate",
            "fixed-evaluate-condition",
            "fixed-parallel",
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SCAFFOLD_ROOT / "config.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", help="output subdirectory below scaffold/outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=2,
        help="maximum simultaneous fixed-condition workers (default: 2)",
    )
    parser.add_argument(
        "--condition",
        help="single fixed-sweep row for sampling or evaluation",
    )
    args = parser.parse_args()
    condition_commands = {"fixed-condition", "fixed-evaluate-condition"}
    if args.command in condition_commands and args.condition is None:
        parser.error(f"{args.command} requires --condition")
    if args.command not in condition_commands and args.condition is not None:
        parser.error("--condition is only valid with a single-condition command")
    if args.parallel_workers < 1:
        parser.error("--parallel-workers must be positive")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
