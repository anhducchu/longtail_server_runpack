from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from experiment import load_experiment


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("FID", "KID", "IS", "F_8", "F_1_8", "ImprovedPrecision", "Recall")


def load_report_module():
    spec = importlib.util.spec_from_file_location(
        "report_native_cifar", ROOT / "tools/report_native_cifar.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_experiment(tmp_path: Path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("LTX_RUNS_ROOT", str(runs_root))
    config = tmp_path / "native.yaml"
    config.write_text(json.dumps({
        "experiment": {
            "name": "native-cifar-test",
            "protocol": "native_cifar_v1",
            "seeds": [0, 1, 2],
        },
        "repository": {"directory": "CCUA-DDPM"},
        "dataset": {"name": "cifar100lt_if100", "root": str(tmp_path / "data")},
        "train": {},
        "evaluation": {},
        "methods": [{"name": "ddpm", "objective": "ddpm"}],
    }), encoding="utf-8")
    return config, load_experiment(config)


def native_metrics(seed: int) -> dict[str, float]:
    return {
        "FID": 10.0 + seed,
        "KID": 0.001 + seed / 1000,
        "IS": 5.0 + seed,
        "F_8": 0.50 + seed / 100,
        "F_1_8": 0.40 + seed / 100,
        "ImprovedPrecision": 0.60 + seed / 100,
        "Recall": 0.70 + seed / 100,
    }


def write_run(task, payload: dict, *, per_class: dict | None = None, success: bool = True) -> None:
    run_dir = Path(task.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if success:
        (run_dir / "SUCCESS").write_text("ok\n", encoding="utf-8")
    (run_dir / "metrics.native.json").write_text(json.dumps(payload), encoding="utf-8")
    if per_class is not None:
        (run_dir / "metrics.per_class.native.json").write_text(
            json.dumps(per_class), encoding="utf-8"
        )


def test_native_report_aggregates_canonical_metrics_across_three_seeds(tmp_path, monkeypatch):
    config, experiment = write_experiment(tmp_path, monkeypatch)
    for run in experiment.runs:
        metrics = native_metrics(run.seed)
        write_run(run, {"metrics": metrics}, per_class={"groups": {
            group: {"FID": metrics["FID"] + offset}
            for group, offset in (("Many", 0.1), ("Medium", 0.2), ("Few", 0.3))
        }})

    output = tmp_path / "report"
    module = load_report_module()
    monkeypatch.setattr("sys.argv", ["report", "--config", str(config), "--output", str(output)])
    assert module.main() == 0

    for filename in ("per_seed.csv", "summary.json", "results.log"):
        assert (output / filename).is_file()
    with (output / "per_seed.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["complete", "complete", "complete"]
    assert {row["per_class_status"] for row in rows} == {"complete"}

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    aggregate = summary["aggregate"]
    assert len(aggregate) == 1
    assert aggregate[0]["complete"] is True
    assert aggregate[0]["completed"] == aggregate[0]["expected"] == 3
    assert aggregate[0]["FID_mean"] == pytest.approx(11.0)
    assert aggregate[0]["FID_std"] == pytest.approx(1.0)
    assert {row["group"] for row in summary["per_class_aggregate"]} == {"Many", "Medium", "Few"}
    assert "complete cells: 1/1" in (output / "results.log").read_text(encoding="utf-8")


def test_native_report_fails_closed_for_missing_success_or_required_metric(tmp_path, monkeypatch):
    config, experiment = write_experiment(tmp_path, monkeypatch)
    for run in experiment.runs:
        metrics = native_metrics(run.seed)
        if run.seed == 1:
            metrics.pop("Recall")
        write_run(run, {"metrics": metrics}, success=run.seed != 2)

    output = tmp_path / "report"
    module = load_report_module()
    monkeypatch.setattr("sys.argv", ["report", "--config", str(config), "--output", str(output)])
    assert module.main() == 2

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    aggregate = summary["aggregate"][0]
    assert aggregate["complete"] is False
    assert aggregate["FID_mean"] is None
    per_seed = {row["seed"]: row for row in summary["per_seed"]}
    assert "missing metrics: Recall" in per_seed[1]["failure_reason"]
    assert "missing SUCCESS" in per_seed[2]["failure_reason"]


def test_native_report_requires_complete_many_medium_few(tmp_path, monkeypatch):
    config, experiment = write_experiment(tmp_path, monkeypatch)
    for run in experiment.runs:
        groups = {"Many": {"FID": 10.0}, "Medium": {"FID": 11.0}, "Few": {"FID": 12.0}}
        if run.seed == 1:
            groups.pop("Few")
        write_run(run, {"metrics": native_metrics(run.seed)}, per_class={"groups": groups})

    output = tmp_path / "report"
    module = load_report_module()
    monkeypatch.setattr("sys.argv", ["report", "--config", str(config), "--output", str(output)])
    assert module.main() == 2

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    failed = next(row for row in summary["per_seed"] if row["seed"] == 1)
    assert failed["status"] == "MISSING"
    assert failed["per_class_status"] == "incomplete"
    assert "missing groups: Few" in failed["failure_reason"]


def test_native_report_accepts_the_seed_zero_baseline_config(tmp_path, monkeypatch):
    config, experiment = write_experiment(tmp_path, monkeypatch)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["experiment"]["seeds"] = [0]
    config.write_text(json.dumps(payload), encoding="utf-8")
    experiment = load_experiment(config)
    write_run(experiment.runs[0], {"metrics": native_metrics(0)}, per_class={"groups": {
        "Many": {"FID": 10.0}, "Medium": {"FID": 11.0}, "Few": {"FID": 12.0},
    }})

    output = tmp_path / "report"
    module = load_report_module()
    monkeypatch.setattr("sys.argv", ["report", "--config", str(config), "--output", str(output)])

    assert module.main() == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["aggregate"][0]["expected"] == 1
    assert summary["aggregate"][0]["FID_std"] == 0.0
