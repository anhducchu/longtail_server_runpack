from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ltx.adapters.ccua import CCUAAdapter
from ltx.config import load_campaign
from ltx.preflight import _check_native_cifar_contract
from ltx.worker import wandb_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/native_cifar100_if100.yaml"


def _campaign():
    return load_campaign(CONFIG)


def _phase_command(phase) -> str:
    return " ".join(str(value) for value in phase.command)


def test_native_contract_uses_one_ccua_host_for_all_objectives():
    campaign = _campaign()

    assert len(campaign.tasks) == 9
    assert {task.method for task in campaign.tasks} == {"ddpm", "cbdm", "ccua"}
    assert {task.seed for task in campaign.tasks} == {0, 1, 2}
    assert {task.adapter for task in campaign.tasks} == {"ccua"}
    assert {task.repository["directory"] for task in campaign.tasks} == {"CCUA-DDPM"}
    assert {task.stage for task in campaign.tasks} == {"c100_if100_core"}
    assert campaign.raw["native_contract"]["adapter"] == "ccua"
    assert campaign.raw["native_contract"]["repository"] == "ccua"


@pytest.mark.parametrize("method", ["ddpm", "cbdm", "ccua"])
def test_ccua_adapter_dispatches_native_objective_and_uses_official_sampler(tmp_path, method):
    campaign = _campaign()
    task = next(task for task in campaign.tasks if task.method == method and task.seed == 0)
    task = replace(task, run_dir=str(tmp_path / method))

    phases = CCUAAdapter(ROOT).phases(task)
    train = phases[0].command
    sample = _phase_command(next(phase for phase in phases if phase.name == "eval"))

    assert train[0] == "python"
    assert train[1:3] == ["main.py", "--train"]
    assert "--notransfer_x0" in train
    if method == "cbdm":
        assert "--cbdm" in train
        assert "--nocbdm" not in train
        assert "--cb_tau=1.0" in train
        assert "--tau=1.0" not in train
    else:
        assert "--cbdm" not in train
        assert "--nocbdm" in train
    if method in {"ddpm", "cbdm"}:
        assert "--ccua_al=0.0" in train
        assert "--ccua_ucl=0.0" in train
    elif method == "ccua":
        assert "--ccua_al=1.0" in train
        assert "--ccua_ucl=1.0" in train
    # Sampling is delegated to CCUA-DDPM's main.py, which constructs its
    # GaussianDiffusionSamplerDDIM. The shared metric evaluator is a later
    # phase and must not replace this sampler command.
    assert "main.py --sample" in sample
    assert "evaluate.py" not in sample
    assert "--batch_size=64" in sample
    assert "--sample_batch_size=128" in sample
    assert "--sample_method=ddim" in sample
    assert "--ddim_skip_step=10" in sample
    assert "--uniform_labels" in sample


def test_ccua_adapter_adds_cuda_preview_and_shared_feature_cache(tmp_path):
    task = next(task for task in _campaign().tasks if task.method == "ddpm" and task.seed == 0)
    task = replace(task, run_dir=str(tmp_path / "ddpm"))
    phases = CCUAAdapter(ROOT).phases(task)

    fast = next(phase for phase in phases if phase.name == "fast_metrics")
    canonical = next(phase for phase in phases if phase.name == "paper_metrics")
    fast_command = _phase_command(fast)
    canonical_command = _phase_command(canonical)

    assert "--mode gpu_fast" in fast_command
    assert "fast_metrics_preview.json" in fast_command
    assert "fast_per_class_fids.json" in fast_command
    assert "--generated-features-cache" in fast_command
    assert "--generated-features-cache" in canonical_command
    assert "--mode detailed" in canonical_command
    assert "--legacy-prd-repeat" not in canonical_command
    # The preview filename is intentionally outside metrics*.json so W&B's
    # canonical collector cannot overwrite the paper artifact with partial data.
    assert "metrics.native.json" not in fast_command


def test_ccua_adapter_dispatches_t2h_transfer_on_the_same_host(tmp_path):
    original = next(task for task in _campaign().tasks if task.method == "ddpm" and task.seed == 0)
    task = replace(
        original,
        method="t2h",
        method_config={"name": "t2h", "objective": "t2h"},
        run_dir=str(tmp_path / "t2h"),
    )

    train = CCUAAdapter(ROOT).phases(task)[0].command
    assert "--transfer_x0" in train
    assert "--transfer_mode=t2h" in train
    assert "--notransfer_x0" not in train
    assert "--nocbdm" in train
    assert "--ccua_al=0.0" in train
    assert "--ccua_ucl=0.0" in train
    unsupported = replace(
        task,
        method="unsupported",
        method_config={"name": "unsupported", "objective": "unsupported"},
        run_dir=str(tmp_path / "unsupported"),
    )
    with pytest.raises(ValueError, match="supports only objective=ddpm, cbdm, t2h, or ccua"):
        CCUAAdapter(ROOT).phases(unsupported)


def test_native_preflight_accepts_ccua_contract_and_rejects_host_or_sampler_drift():
    campaign = _campaign()
    checks = _check_native_cifar_contract(campaign)
    wanted = {"native-cifar-matrix", "native-cifar-adapters", "native-cifar-objectives",
              "native-cifar-repository", "native-cifar-controls"}
    assert {check.name for check in checks} >= wanted
    assert not [check for check in checks if check.level == "ERROR"], checks

    first = campaign.tasks[0]
    campaign.tasks[0] = replace(first, adapter="unsupported")
    drift_checks = _check_native_cifar_contract(campaign)
    assert next(check for check in drift_checks if check.name == "native-cifar-adapters").level == "ERROR"

    campaign = _campaign()
    first = campaign.tasks[0]
    campaign.tasks[0] = replace(first, eval={**first.eval, "sample_method": "ddpm"})
    drift_checks = _check_native_cifar_contract(campaign)
    controls = next(check for check in drift_checks if check.name == "native-cifar-controls")
    assert controls.level == "ERROR"
    assert "official DDIM" in controls.message


def test_ccua_adapter_resumes_an_explicit_full_state_checkpoint(tmp_path):
    campaign = _campaign()
    original = next(task for task in campaign.tasks if task.method == "ddpm" and task.seed == 0)
    external = tmp_path / "ccua_source" / "ckpt_250000.pt"
    external.parent.mkdir(parents=True)
    external.touch()
    task = replace(
        original,
        run_dir=str(tmp_path / "destination"),
        method_config={
            **original.method_config,
            "resume_checkpoint": str(external),
            "resume_mode": "full",
        },
    )

    train = CCUAAdapter(ROOT).phases(task)[0].command
    assert "--resume" in train
    assert f"--resume_dir={external.parent}" in train
    assert "--ckpt_step=250000" in train
    assert "--total_steps=50001" in train


def test_ccua_adapter_does_not_discard_a_step_zero_checkpoint(tmp_path):
    campaign = _campaign()
    original = next(task for task in campaign.tasks if task.method == "ddpm" and task.seed == 0)
    run_dir = tmp_path / "destination"
    run_dir.mkdir()
    (run_dir / "ckpt_0.pt").touch()
    task = replace(original, run_dir=str(run_dir))

    train = CCUAAdapter(ROOT).phases(task)[0].command
    assert "--resume" in train
    assert f"--resume_dir={run_dir}" in train
    assert "--ckpt_step=0" in train
    assert "--total_steps=300001" in train


@pytest.mark.parametrize("method", ["ddpm", "cbdm", "ccua"])
def test_wandb_config_records_the_ccua_objective_branch(method):
    task = next(task for task in _campaign().tasks if task.method == method and task.seed == 0)
    config = wandb_config(task)
    assert config["upstream_repo"] == "CCUA-DDPM"
    assert config["objective/objective"] == method
