from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from scaffold.core import (
    ScalarScaffold,
    alpha_bar_schedule,
    effective_class_weights,
    forward_path_pair,
    predict_epsilon,
    reverse_sample,
)
from scaffold.common import OUTPUTS_ROOT, load_frozen_score, output_directory
from scaffold.sample import (
    FIXED_SWEEP,
    fixed_sweep_conditions,
    generate_condition,
    write_fixed_sweep_table,
)
from scaffold.train import training_signature
from scaffold.run import run_fixed_evaluate_parallel, run_fixed_parallel


class TinyFrozenScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1), requires_grad=False)

    def forward(self, images, times, y=None):
        del times, y
        return images * self.scale


def test_scaffold_starts_at_native_rho_one():
    scaffold = ScalarScaffold(hidden_width=8, rho_min=0.05, rho_max=2.0)
    rho = scaffold(torch.linspace(0, 1, 7), torch.linspace(0, 1, 7))

    assert torch.allclose(rho, torch.ones_like(rho), atol=1e-6)


def test_frozen_score_has_no_parameter_gradient_but_phi_does():
    model = TinyFrozenScore().eval()
    scaffold = ScalarScaffold(hidden_width=8)
    alpha_bar = alpha_bar_schedule(8, 1e-4, 0.02, device=torch.device("cpu"))
    images = torch.randn(2, 3, 4, 4)
    labels = torch.tensor([0, 1])
    counts = torch.tensor([10.0, 2.0])

    paths = forward_path_pair(
        model,
        scaffold,
        images,
        labels,
        counts,
        alpha_bar,
        steps=8,
        stride=2,
        microbatch=2,
        generator=torch.Generator().manual_seed(7),
        guidance_scale=1.5,
    )
    paths.square().mean().backward()

    assert model.scale.grad is None
    assert any(parameter.grad is not None for parameter in scaffold.parameters())


def test_paired_reverse_sampling_is_reproducible():
    model = TinyFrozenScore().eval()
    alpha_bar = alpha_bar_schedule(8, 1e-4, 0.02, device=torch.device("cpu"))
    labels = torch.tensor([0, 1])
    counts = torch.tensor([10.0, 2.0])

    def generate():
        return reverse_sample(
            model,
            None,
            labels,
            counts,
            alpha_bar,
            image_size=4,
            steps=8,
            stride=2,
            microbatch=2,
            generator=torch.Generator().manual_seed(11),
            rho_override=1.0,
            guidance_scale=1.5,
        )

    assert torch.equal(generate(), generate())


def test_pf_ode_matches_vendor_ddim_update():
    model = TinyFrozenScore().eval()
    alpha_bar = alpha_bar_schedule(8, 1e-4, 0.02, device=torch.device("cpu"))
    labels = torch.tensor([0, 1])
    counts = torch.tensor([10.0, 2.0])
    seed = 13
    actual = reverse_sample(
        model,
        None,
        labels,
        counts,
        alpha_bar,
        image_size=4,
        steps=8,
        stride=2,
        microbatch=2,
        generator=torch.Generator().manual_seed(seed),
        rho_override=0.0,
        guidance_scale=0.0,
    )

    expected = torch.randn(
        (2, 3, 4, 4), generator=torch.Generator().manual_seed(seed)
    )
    for current_step in (6, 4, 2):
        next_step = current_step - 2
        current_alpha = alpha_bar[current_step]
        next_alpha = alpha_bar[next_step]
        epsilon = expected * model.scale
        predicted_clean = (
            expected - (1.0 - current_alpha).sqrt() * epsilon
        ) / current_alpha.sqrt()
        expected = (
            next_alpha.sqrt() * predicted_clean
            + (1.0 - next_alpha).sqrt() * epsilon
        )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_guided_epsilon_matches_vendor_cfg_formula():
    class ConditionalScore(nn.Module):
        def forward(self, images, times, y=None):
            del times
            value = -1.0 if y is None else 1.0
            return torch.full_like(images, value)

    images = torch.zeros(2, 3, 2, 2)
    epsilon = predict_epsilon(
        ConditionalScore(),
        images,
        torch.zeros(2, dtype=torch.long),
        torch.tensor([0, 1]),
        microbatch=1,
        preserve_input_gradient=False,
        guidance_scale=1.5,
    )

    # Vendor CCUA-DDPM uses eps_cond + omega * (eps_cond - eps_uncond).
    assert torch.equal(epsilon, torch.full_like(images, 4.0))


def test_generated_samples_are_saved_in_evaluator_zero_one_range(tmp_path, monkeypatch):
    raw = torch.linspace(-1.0, 1.0, 24).reshape(2, 3, 2, 2)

    def fake_reverse_sample(*_args, **kwargs):
        assert kwargs["guidance_scale"] == 1.5
        return raw

    monkeypatch.setattr("scaffold.sample.reverse_sample", fake_reverse_sample)
    config = {
        "dataset": {"image_size": 2},
        "diffusion": {"steps": 8},
        "model": {"microbatch": 2},
        "sampling": {
            "num_images": 2,
            "batch_size": 2,
            "stride": 2,
            "seed": 7,
            "guidance_scale": 1.5,
        },
    }

    destination = generate_condition(
        condition="standard_sde",
        model=nn.Identity(),
        scaffold=None,
        rho_override=1.0,
        counts=torch.tensor([1.0, 1.0]),
        labels=np.array([0, 1]),
        alpha_bar=torch.ones(8),
        source_checkpoint=tmp_path / "checkpoint.pt",
        source_sha256="abc",
        phi_sha256=None,
        config=config,
        output=tmp_path,
        device=torch.device("cpu"),
    )

    saved = np.load(destination)
    np.testing.assert_allclose(saved, ((raw + 1.0) / 2.0).numpy())
    assert float(saved.min()) == 0.0
    assert float(saved.max()) == 1.0


def test_effective_weights_are_normalized():
    weights = effective_class_weights(torch.tensor([100.0, 10.0, 1.0]), 0.5)
    assert float(weights.mean()) == pytest.approx(1.0)


def test_output_cannot_escape_scaffold_directory():
    config = {"experiment": {"name": "safe"}}
    assert output_directory(config, None) == (OUTPUTS_ROOT / "safe").resolve()
    with pytest.raises(ValueError):
        output_directory(config, "../outside")


def test_training_signature_allows_only_longer_resume():
    config = {
        "dataset": {"name": "cifar"},
        "diffusion": {"steps": 1000},
        "model": {"channels": 128},
        "training": {"updates": 10, "save_every": 5, "log_every": 1, "learning_rate": 1e-3},
        "sampling": {"guidance_scale": 1.5},
    }
    extended = {**config, "training": {**config["training"], "updates": 20}}
    changed = {**config, "training": {**config["training"], "learning_rate": 1e-4}}

    assert training_signature(config) == training_signature(extended)
    assert training_signature(config) != training_signature(changed)


def test_loads_ema_checkpoint_and_freezes_every_score_parameter(tmp_path):
    from model.model import UNet

    score = UNet(
        T=8,
        ch=32,
        ch_mult=[1],
        attn=[0],
        num_res_blocks=1,
        dropout=0.0,
        cond=True,
        num_class=2,
    )
    checkpoint = tmp_path / "score.pt"
    torch.save({"ema_model": score.state_dict()}, checkpoint)
    config = {
        "dataset": {"num_classes": 2},
        "diffusion": {"steps": 8},
        "model": {
            "checkpoint_key": "ema_model",
            "channels": 32,
            "channel_multipliers": [1],
            "attention_levels": [0],
            "residual_blocks": 1,
            "dropout": 0.0,
        },
    }

    loaded = load_frozen_score(checkpoint, config, torch.device("cpu"))

    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_fixed_sweep_has_the_requested_five_ratios():
    assert [rho for _name, _label, rho in FIXED_SWEEP] == [0.0, 0.25, 0.5, 1.0, 2.0]


def test_fixed_sweep_can_select_one_parallel_worker_condition():
    assert fixed_sweep_conditions("reduced_05") == (("reduced_05", "Reduced", 0.5),)
    with pytest.raises(ValueError, match="unknown fixed-sweep condition"):
        fixed_sweep_conditions("missing")


def test_fixed_parallel_limits_active_workers_to_two(tmp_path, monkeypatch):
    active = 0
    maximum = 0
    started = []

    class FakeProcess:
        def __init__(self, condition):
            nonlocal active, maximum
            self.pid = 1000 + len(started)
            self.returncode = None
            self.condition = condition
            active += 1
            maximum = max(maximum, active)

        def wait(self):
            nonlocal active
            active -= 1
            self.returncode = 0
            return 0

    def fake_popen(command, **_kwargs):
        condition = command[command.index("--condition") + 1]
        started.append(condition)
        return FakeProcess(condition)

    evaluated = []
    monkeypatch.setattr("scaffold.run.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "scaffold.run.subprocess.run",
        lambda command, **_kwargs: evaluated.append(command),
    )

    run_fixed_parallel(
        SimpleNamespace(device="cuda", parallel_workers=2),
        tmp_path / "config.yaml",
        tmp_path / "checkpoint.pt",
        tmp_path / "sweep",
    )

    assert maximum == 2
    assert started == [condition for condition, _label, _rho in FIXED_SWEEP]
    assert "fixed-evaluate" in evaluated[0]


def test_fixed_evaluation_limits_active_workers_to_two(tmp_path, monkeypatch):
    active = 0
    maximum = 0
    started = []

    class FakeProcess:
        def __init__(self):
            nonlocal active, maximum
            self.pid = 2000 + len(started)
            active += 1
            maximum = max(maximum, active)

        def wait(self):
            nonlocal active
            active -= 1
            return 0

    def fake_popen(command, **_kwargs):
        started.append(command[command.index("--condition") + 1])
        return FakeProcess()

    monkeypatch.setattr("scaffold.run.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "scaffold.run.write_fixed_sweep_table",
        lambda output: output / "fixed_sweep_results.md",
    )

    result = run_fixed_evaluate_parallel(
        SimpleNamespace(parallel_workers=2),
        tmp_path / "config.yaml",
        tmp_path / "checkpoint.pt",
        tmp_path / "sweep",
    )

    assert maximum == 2
    assert started == [condition for condition, _label, _rho in FIXED_SWEEP]
    assert result.name == "fixed_sweep_results.md"


def test_fixed_sweep_table_reads_canonical_metric_artifacts(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    for index, (condition, _label, _rho) in enumerate(FIXED_SWEEP):
        (metrics / f"{condition}.json").write_text(
            '{"metrics": {"FID": %s, "Recall": 0.5}}' % (10 + index),
            encoding="utf-8",
        )
        (metrics / f"{condition}.per_class.json").write_text(
            '{"groups": {"Many": {"FID": 1}, "Medium": {"FID": 2}, "Few": {"FID": 3}}}',
            encoding="utf-8",
        )

    table = write_fixed_sweep_table(tmp_path)

    assert "Standard SDE" in table.read_text(encoding="utf-8")
    assert "N/A" in table.read_text(encoding="utf-8")
