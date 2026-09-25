from pathlib import Path
import subprocess
import sys

import pytest

from runner.config import apply_run_overrides, load_campaign


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/native_cifar100_if100.yaml"


def test_main_script_exposes_the_cli():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for command in ("plan", "preflight", "run", "status", "retry-failed", "stop"):
        assert command in result.stdout


def test_run_help_explains_its_overrides():
    result = subprocess.run(
        [sys.executable, "main.py", "run", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "campaign YAML" in result.stdout
    assert "concurrent tasks per GPU" in result.stdout
    assert "checkpoint path" in result.stdout


def test_run_overrides_machine_limits():
    campaign = load_campaign(CONFIG)
    apply_run_overrides(campaign, gpus="1,3", per_gpu=2, jobs="auto")

    assert campaign.server["machine"]["gpu_ids"] == [1, 3]
    assert campaign.server["machine"]["tasks_per_gpu"] == 2
    assert campaign.server["machine"]["max_concurrent"] == "auto"


def test_run_override_targets_one_resume_task(tmp_path):
    campaign = load_campaign(CONFIG)
    template = str(tmp_path / "checkpoint_{seed}.pt")
    apply_run_overrides(
        campaign,
        resume_checkpoint=template,
        resume_method="ddpm",
        resume_seed=1,
        resume_step=25_000,
    )

    selected = [task for task in campaign.tasks if "resume_checkpoint" in task.method_config]
    assert len(selected) == 1
    assert selected[0].method == "ddpm"
    assert selected[0].seed == 1
    assert selected[0].method_config["resume_checkpoint"] == str(
        (tmp_path / "checkpoint_1.pt").resolve()
    )
    assert selected[0].method_config["resume_step"] == 25_000


def test_resume_selectors_require_a_checkpoint():
    campaign = load_campaign(CONFIG)
    with pytest.raises(ValueError, match="require --resume-checkpoint"):
        apply_run_overrides(campaign, resume_method="ddpm")
