#!/usr/bin/env python3
"""Entrypoint for the isolated frozen-score scaffold experiment.

Flow: load EMA score -> freeze theta -> train phi -> paired sample -> evaluate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scaffold.common import (
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
    FIXED_SWEEP,
    evaluate_conditions,
    sample_conditions,
    sample_fixed_sweep,
    write_fixed_sweep_table,
)
from scaffold.train import train_scaffold


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config.resolve())
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
        fixed_conditions = tuple(condition for condition, _label, _rho in FIXED_SWEEP)
        evaluate_conditions(config, output, fixed_conditions)
        result = write_fixed_sweep_table(output)
        print(f"[result] {result}")
    if args.command == "all" and bool(config["evaluation"].get("enabled", False)):
        evaluate_conditions(config, output)


def show_plan(config: dict, checkpoint: Path, output: Path) -> None:
    condition_count = 5 if config["experiment"].get("kind") == "fixed_sweep" else 2
    print(f"score checkpoint: {checkpoint}")
    print(f"frozen weights:   {config['model']['checkpoint_key']}")
    print(f"output:           {output}")
    print(f"updates:          {config['training']['updates']}")
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
        choices=("plan", "train", "sample", "evaluate", "all", "fixed-sweep"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SCAFFOLD_ROOT / "config.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", help="output subdirectory below scaffold/outputs")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
