from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import main as cli
from experiment import load_experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/native_cifar100_if100.yaml"


def run_cli(*arguments: str):
    return subprocess.run(
        [sys.executable, "main.py", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_main_exposes_only_the_direct_workflow():
    result = run_cli("--help")

    assert result.returncode == 0, result.stderr
    for command in ("plan", "preflight", "run"):
        assert command in result.stdout
    for retired in ("status", "retry-failed", "stop"):
        assert retired not in result.stdout


def test_run_help_explains_selection_and_stages():
    result = run_cli("run", "--help")

    assert result.returncode == 0, result.stderr
    assert "--method" in result.stdout
    assert "--seed" in result.stdout
    assert "--index" in result.stdout
    assert "--stage" in result.stdout
    assert "--resume-checkpoint" in result.stdout
    assert "per-gpu" not in result.stdout


def test_plan_prints_the_nine_stable_array_indices():
    result = run_cli("plan", "--config", str(CONFIG))

    assert result.returncode == 0, result.stderr
    assert "(9 runs)" in result.stdout
    assert result.stdout.count("method=") == 9
    assert "[00] method=ddpm" in result.stdout
    assert "[08] method=ccua" in result.stdout


def test_one_run_is_train_then_sample_then_evaluate(tmp_path, monkeypatch):
    run = replace(load_experiment(CONFIG).runs[0], run_dir=tmp_path / "run")
    calls = []
    sample_sets = [object()]
    monkeypatch.setattr(cli, "train", lambda *_args, **_kwargs: calls.append("train"))
    monkeypatch.setattr(
        cli,
        "sample",
        lambda *_args, **_kwargs: calls.append("sample") or sample_sets,
    )
    monkeypatch.setattr(
        cli,
        "evaluate",
        lambda _run, samples, **_kwargs: calls.append(("evaluate", samples)),
    )

    cli.run_one(run)

    assert calls == ["train", "sample", ("evaluate", sample_sets)]
    assert (run.run_dir / "config.resolved.json").is_file()
    assert (run.run_dir / "SUCCESS").is_file()
