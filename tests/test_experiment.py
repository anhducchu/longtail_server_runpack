from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import experiment as runtime


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/native_cifar100_if100.yaml"
SEED0_CONFIG = ROOT / "configs/native_cifar100_if100_seed0.yaml"


def load_runs():
    return runtime.load_experiment(CONFIG)


def run_for(method: str, seed: int = 0):
    return next(
        run for run in load_runs().runs
        if run.method == method and run.seed == seed
    )


def test_config_expands_to_nine_direct_runs():
    experiment = load_runs()

    assert len(experiment.runs) == 9
    assert {run.method for run in experiment.runs} == {"ddpm", "cbdm", "ccua"}
    assert {run.seed for run in experiment.runs} == {0, 1, 2}
    assert {run.repo_dir.name for run in experiment.runs} == {"CCUA-DDPM"}


@pytest.mark.parametrize("method", ["ddpm", "cbdm", "ccua"])
def test_train_command_selects_the_objective(method, tmp_path):
    run = replace(run_for(method), run_dir=tmp_path / method)
    command = runtime.train_command(run)

    assert command[1:3] == ["main.py", "--train"]
    assert "--notransfer_x0" in command
    if method == "cbdm":
        assert "--cbdm" in command
        assert "--nocbdm" not in command
        assert "--cb_tau=1.0" in command
    else:
        assert "--cbdm" not in command
        assert "--nocbdm" in command
    if method in {"ddpm", "cbdm"}:
        assert "--ccua_al=0.0" in command
        assert "--ccua_ucl=0.0" in command
    else:
        assert "--ccua_al=1.0" in command
        assert "--ccua_ucl=1.0" in command


def test_t2h_uses_the_transfer_target_on_the_same_trainer(tmp_path):
    experiment = runtime.load_experiment(SEED0_CONFIG)
    run = next(run for run in experiment.runs if run.method == "t2h")
    command = runtime.train_command(replace(run, run_dir=tmp_path / "t2h"))

    assert "--transfer_x0" in command
    assert "--notransfer_x0" not in command
    assert "--nocbdm" in command
    assert "--ccua_al=0.0" in command
    assert "--ccua_ucl=0.0" in command


def test_sample_calls_the_official_vendor_sampler(tmp_path, monkeypatch):
    run = replace(run_for("ddpm"), run_dir=tmp_path / "ddpm")
    calls = []
    monkeypatch.setattr(
        runtime,
        "_run_command",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    outputs = runtime.sample(run, gpu=2)
    command, options = calls[0]

    assert command[1:3] == ["main.py", "--sample"]
    assert "--sample_method=ddim" in command
    assert "--ddim_skip_step=10" in command
    assert "--sample_batch_size=128" in command
    assert "--uniform_labels" in command
    assert options == {"cwd": run.repo_dir, "gpu": 2}
    assert outputs[0].images.name == "ccua_samples.npy"


def test_evaluation_has_preview_and_canonical_outputs(tmp_path, monkeypatch):
    run = replace(run_for("ddpm"), run_dir=tmp_path / "ddpm")
    calls = []
    monkeypatch.setattr(
        runtime,
        "_run_command",
        lambda command, **kwargs: calls.append(command),
    )

    runtime.evaluate(run, runtime.expected_samples(run))
    rendered = [" ".join(command) for command in calls]

    assert any("--mode gpu_fast" in command for command in rendered)
    assert any("fast_metrics_preview.json" in command for command in rendered)
    assert any("--mode detailed" in command for command in rendered)
    assert any("metrics.native.json" in command for command in rendered)
    assert all("--generated-features-cache" in command for command in rendered)


def test_explicit_checkpoint_resumes_full_training_state(tmp_path):
    external = tmp_path / "source" / "ckpt_250000.pt"
    external.parent.mkdir()
    external.touch()
    run = replace(run_for("ddpm"), run_dir=tmp_path / "destination")

    command = runtime.train_command(run, resume_checkpoint=str(external))

    assert "--resume" in command
    assert f"--resume_dir={external.parent}" in command
    assert "--ckpt_step=250000" in command
    assert "--total_steps=50001" in command


def test_local_step_zero_checkpoint_is_not_discarded(tmp_path):
    run_dir = tmp_path / "destination"
    run_dir.mkdir()
    (run_dir / "ckpt_0.pt").touch()
    command = runtime.train_command(replace(run_for("ddpm"), run_dir=run_dir))

    assert "--resume" in command
    assert "--ckpt_step=0" in command
    assert "--total_steps=300001" in command


def test_slurm_index_selects_exactly_one_run():
    experiment = load_runs()
    selected = runtime.select_runs(experiment, index=4)

    assert len(selected) == 1
    assert selected[0] == experiment.runs[4]
    with pytest.raises(ValueError, match="cannot be combined"):
        runtime.select_runs(experiment, index=0, method="ddpm")


def test_preflight_detects_scientific_protocol_drift():
    experiment = load_runs()
    first = experiment.runs[0]
    drifted = replace(first, train={**first.train, "batch_size": 32})
    changed = replace(experiment, runs=[drifted, *experiment.runs[1:]])

    check = runtime._protocol_check(changed)
    assert check.level == "ERROR"
    assert "train.batch_size" in check.message
