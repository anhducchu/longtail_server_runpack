#!/usr/bin/env python3
"""Write a fail-closed report for the native CIFAR protocol."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from runner.config import LoadedCampaign, load_campaign
from runner.utils import load_runtime_env


METRICS = ("FID", "KID", "IS", "F_8", "F_1_8", "ImprovedPrecision", "Recall")
GROUPS = ("Many", "Medium", "Few")
METRICS_FILENAME = "metrics.native.json"
PER_CLASS_FILENAME = "metrics.per_class.native.json"


def finite(value: Any) -> float | None:
    """Convert a finite scalar, returning ``None`` for absent/invalid input."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def read_metrics(run_dir: Path) -> dict[str, float]:
    """Read only the native main-metrics artifact; never use legacy fallbacks."""
    path = run_dir / METRICS_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    values = payload.get("metrics") if isinstance(payload, dict) else None
    if not isinstance(values, dict):
        return {}
    return {
        metric: value
        for metric in METRICS
        if (value := finite(values.get(metric))) is not None
    }


def read_per_class_metrics(run_dir: Path) -> tuple[str, dict[str, float]]:
    """Return the optional native group FIDs and an auditable artifact status."""
    path = run_dir / PER_CLASS_FILENAME
    if not path.is_file():
        return "absent", {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return "invalid", {}
    groups = payload.get("groups") if isinstance(payload, dict) else None
    if not isinstance(groups, dict):
        return "invalid", {}
    parsed = {}
    for group in GROUPS:
        group_value = groups.get(group)
        value = group_value.get("FID") if isinstance(group_value, dict) else group_value
        if (parsed_value := finite(value)) is not None:
            parsed[group] = parsed_value
    return ("complete" if set(parsed) == set(GROUPS) else "incomplete"), parsed


def _missing_metric_reason(metrics: dict[str, float]) -> str:
    missing = [metric for metric in METRICS if metric not in metrics]
    return "" if not missing else f"missing metrics: {', '.join(missing)}"


def _write_per_seed(output: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "dataset", "method", "adapter", "seed", "status", "per_class_status",
        "failure_reason", *METRICS, "run_dir",
    ]
    with (output / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {column: row.get(column) for column in columns}
            for row in sorted(rows, key=lambda row: (row["dataset"], row["method"], row["seed"]))
        )


def _aggregate(rows: list[dict[str, Any]], expected_seeds: list[int]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"])].append(row)
    aggregate = []
    for (dataset, method), group in sorted(grouped.items()):
        complete_rows = [row for row in group if row["status"] == "complete"]
        complete = (
            len(group) == len(expected_seeds)
            and sorted(row["seed"] for row in group) == expected_seeds
            and len(complete_rows) == len(expected_seeds)
        )
        result: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "adapter": sorted({row["adapter"] for row in group}),
            "complete": complete,
            "completed": len(complete_rows),
            "expected": len(expected_seeds),
            "seed_values": {
                metric: {row["seed"]: row[metric] for row in complete_rows}
                for metric in METRICS
            },
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in complete_rows] if complete else []
            result[f"{metric}_mean"], result[f"{metric}_std"] = mean_std(values)
        aggregate.append(result)
    return aggregate


def _aggregate_per_class(rows: list[dict[str, Any]], expected_seeds: list[int]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for group, fid in row["per_class"].items():
            grouped[(row["dataset"], row["method"], group)].append({"seed": row["seed"], "FID": fid})
    aggregate = []
    for (dataset, method, group), values in sorted(grouped.items()):
        by_seed = {row["seed"]: row["FID"] for row in values}
        complete = sorted(by_seed) == expected_seeds
        mean, std = mean_std(list(by_seed.values())) if complete else (None, None)
        aggregate.append({
            "dataset": dataset,
            "method": method,
            "group": group,
            "complete": complete,
            "completed": len(by_seed),
            "expected": len(expected_seeds),
            "seed_values": by_seed,
            "FID_mean": mean,
            "FID_std": std,
        })
    return aggregate


def _summary_payload(campaign: LoadedCampaign, rows: list[dict[str, Any]], aggregate: list[dict[str, Any]],
                     per_class_aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    complete = all(row["complete"] for row in aggregate)
    return {
        "protocol": "native_cifar_v1",
        "claim_status": "NATIVE_CIFAR_V1_COMPLETE" if complete else "NATIVE_CIFAR_V1_INCOMPLETE",
        "campaign": campaign.raw.get("campaign", {}),
        "per_seed": rows,
        "aggregate": aggregate,
        "per_class_aggregate": per_class_aggregate,
    }


def _write_results_log(output: Path, payload: dict[str, Any]) -> None:
    aggregate = payload["aggregate"]
    incomplete = [row for row in aggregate if not row["complete"]]
    lines = [
        "# Native CIFAR v1 results log",
        f"generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"protocol: {payload['protocol']}",
        f"claim_status: {payload['claim_status']}",
        "",
        "## Per-seed status",
    ]
    for row in sorted(payload["per_seed"], key=lambda item: (item["dataset"], item["method"], item["seed"])):
        reason = row["failure_reason"] or "ok"
        lines.append(
            f"{row['dataset']} {row['method']} seed={row['seed']}: {row['status']} "
            f"(per_class={row['per_class_status']}; {reason})"
        )
    lines.extend(["", "## Seed aggregate"])
    for row in aggregate:
        values = " | ".join(
            f"{metric}=" + (
                "MISSING" if row[f"{metric}_mean"] is None
                else f"{row[f'{metric}_mean']:.6g} ± {row[f'{metric}_std']:.6g}"
            )
            for metric in METRICS
        )
        lines.append(
            f"{row['dataset']} {row['method']}: {row['completed']}/{row['expected']} "
            f"{'COMPLETE' if row['complete'] else 'INCOMPLETE'} | {values}"
        )
    lines.extend(["", f"complete cells: {len(aggregate) - len(incomplete)}/{len(aggregate)}"])
    (output / "results.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _maybe_upload_wandb(campaign: LoadedCampaign, output: Path, rows: list[dict[str, Any]],
                        aggregate: list[dict[str, Any]]) -> None:
    """Best-effort W&B upload. Local reporting and its exit status never depend on it."""
    try:
        import wandb

        runtime = campaign.server.get("runtime", {})
        run = wandb.init(
            project=runtime.get("wandb_project", "longtail"),
            entity=runtime.get("wandb_entity") or None,
            name=f"{campaign.raw['campaign']['name']}-native-report",
            job_type="native-cifar-report",
            mode=runtime.get("wandb_mode", "online"),
            config={"campaign": campaign.raw.get("campaign", {})},
        )
        run.log({"comparison/per_seed": wandb.Table(
            columns=[key for key in rows[0] if key != "per_class"],
            data=[[value for key, value in row.items() if key != "per_class"] for row in rows],
        )})
        run.log({"comparison/aggregate": wandb.Table(
            columns=[key for key in aggregate[0] if key != "seed_values"],
            data=[[value for key, value in row.items() if key != "seed_values"] for row in aggregate],
        )})
        artifact = wandb.Artifact(f"{campaign.raw['campaign']['name']}-native-report", type="evaluation-report")
        for filename in ("per_seed.csv", "summary.json", "results.log"):
            artifact.add_file(str(output / filename))
        run.log_artifact(artifact)
        run.finish(exit_code=0)
    except Exception as exc:
        print(f"[report] W&B upload skipped/failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    campaign = load_campaign(args.config)
    if campaign.raw.get("campaign", {}).get("protocol") != "native_cifar_v1":
        raise ValueError("report_native_cifar.py only accepts protocol=native_cifar_v1")
    expected_seeds = sorted(set(map(int, campaign.raw.get("campaign", {}).get("paired_seeds", []))))
    if not expected_seeds:
        raise ValueError("native_cifar_v1 report requires at least one campaign.paired_seeds entry")
    if not campaign.tasks:
        raise ValueError("native_cifar_v1 report requires at least one task")

    load_runtime_env(campaign.root)
    output = Path(args.output) if args.output else (
        Path(campaign.server["runtime"]["runs_root"]) / campaign.raw["campaign"]["name"] / "report"
    )
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for task in campaign.tasks:
        run_dir = Path(task.run_dir)
        metrics = read_metrics(run_dir)
        per_class_status, per_class = read_per_class_metrics(run_dir)
        reasons = []
        if not (run_dir / "SUCCESS").is_file():
            reasons.append("missing SUCCESS")
        if missing := _missing_metric_reason(metrics):
            reasons.append(missing)
        missing_groups = [group for group in GROUPS if group not in per_class]
        if missing_groups:
            reasons.append(f"missing groups: {', '.join(missing_groups)}")
        rows.append({
            "dataset": task.dataset["name"],
            "method": task.method,
            "adapter": task.adapter,
            "seed": int(task.seed),
            "status": "complete" if not reasons else "MISSING",
            "per_class_status": per_class_status,
            "failure_reason": "; ".join(reasons),
            **{metric: metrics.get(metric) for metric in METRICS},
            "run_dir": str(run_dir),
            "per_class": per_class,
        })

    _write_per_seed(output, rows)
    aggregate = _aggregate(rows, expected_seeds)
    per_class_aggregate = _aggregate_per_class(rows, expected_seeds)
    payload = _summary_payload(campaign, rows, aggregate, per_class_aggregate)
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_results_log(output, payload)
    if args.wandb:
        _maybe_upload_wandb(campaign, output, rows, aggregate)

    complete_cells = sum(row["complete"] for row in aggregate)
    print(f"[report] wrote {output / 'results.log'}; complete cells={complete_cells}/{len(aggregate)}")
    return 0 if complete_cells == len(aggregate) else 2


if __name__ == "__main__":
    raise SystemExit(main())
