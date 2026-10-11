"""Scaffold networks share one interface; the frozen denoiser wrapper follows the vendor CFG rule."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from scaffold.backbone import (
    FixedScaffold, FrozenEpsilon, MLPScaffold, ShuffledStateScaffold, TimeOnlyScaffold, UNetScaffold, build_scaffold,
)

SETTINGS = {
    'mlp': {'kind': 'mlp', 'inputs': 'state_time_class', 'image_size': 8, 'hidden_dims': [16, 16],
            'condition_dim': 8, 'variance_floor': 1e-4, 'variance_initial': 1.0},
    'unet': {'kind': 'unet', 'inputs': 'state_time_class', 'base_channels': 8, 'channel_multipliers': [1, 2],
             'residual_blocks': 1, 'condition_dim': 8, 'variance_floor': 1e-4, 'variance_initial': 0.5},
}


def inputs(batch=6):
    generator = torch.Generator().manual_seed(0)
    return torch.randn(batch, 3, 8, 8, generator=generator), torch.rand(batch, generator=generator), torch.arange(batch) % 4


@pytest.mark.parametrize('kind', ['mlp', 'unet'])
def test_learned_scaffolds_start_at_the_configured_multiplier(kind):
    scaffold = build_scaffold(SETTINGS[kind], num_classes=4)
    assert isinstance(scaffold, MLPScaffold if kind == 'mlp' else UNetScaffold)
    state, time, labels = inputs()
    out = scaffold(state, time, labels)
    assert out.shape == state.shape
    assert torch.allclose(out, torch.full_like(out, SETTINGS[kind]['variance_initial']), atol=1e-6)


@pytest.mark.parametrize('kind', ['mlp', 'unet'])
def test_learned_scaffolds_are_positive_and_use_every_input(kind):
    scaffold = build_scaffold(SETTINGS[kind], num_classes=4)
    nn.init.normal_(scaffold.output.weight, std=0.5)
    state, time, labels = inputs()
    out = scaffold(state, time, labels)
    assert bool((out > SETTINGS[kind]['variance_floor']).all())
    assert not torch.allclose(out, scaffold(state + 1.0, time, labels))
    assert not torch.allclose(out, scaffold(state, 1.0 - time, labels))
    assert not torch.allclose(out, scaffold(state, time, (labels + 1) % 4))
    assert torch.allclose(scaffold(state, 0.3, labels), scaffold(state, torch.full((6,), 0.3), labels))


def test_time_only_scaffold_ignores_the_state():
    scaffold = build_scaffold({**SETTINGS['mlp'], 'inputs': 'time_class'}, num_classes=4)
    assert isinstance(scaffold, TimeOnlyScaffold)
    nn.init.normal_(scaffold.scaffold.output.weight, std=0.5)
    state, time, labels = inputs()
    assert torch.equal(scaffold(state, time, labels), scaffold(torch.randn_like(state), time, labels))
    assert not torch.allclose(scaffold(state, time, labels), scaffold(state, 1.0 - time, labels))


def test_shuffled_state_scaffold_borrows_a_state_of_the_same_class():
    inner = MLPScaffold(4, image_size=8, hidden_dims=(16,), condition_dim=8)
    nn.init.normal_(inner.output.weight, std=0.5)
    state, time, _ = inputs()
    labels = torch.tensor([0, 1, 0, 1, 1, 0])
    donor = torch.tensor([5, 4, 0, 1, 3, 2])  # previous sample of the same class, cyclically
    assert torch.equal(labels[donor], labels) and not torch.equal(donor, torch.arange(6))
    assert torch.equal(ShuffledStateScaffold(inner)(state, time, labels), inner(state[donor], time, labels))
    with pytest.raises(ValueError, match='two samples'):
        ShuffledStateScaffold(inner)(state, time, torch.tensor([0, 0, 1, 1, 2, 0]))


def test_fixed_scaffold_broadcasts_and_validates():
    state, time, labels = inputs()
    assert torch.equal(FixedScaffold(0.25)(state, time, labels), torch.full((6, 1, 1, 1), 0.25))
    with pytest.raises(ValueError):
        FixedScaffold(-1.0)


def test_build_scaffold_rejects_unknown_settings():
    with pytest.raises(ValueError):
        build_scaffold({**SETTINGS['mlp'], 'kind': 'transformer'}, 4)
    with pytest.raises(ValueError):
        build_scaffold({**SETTINGS['mlp'], 'inputs': 'state_only'}, 4)
    with pytest.raises(ValueError):
        build_scaffold({**SETTINGS['mlp'], 'variance_floor': 2.0}, 4)


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, t, y=None):
        shift = 0.0 if y is None else y.float()[:, None, None, None]
        return self.scale * x + shift + t.float()[:, None, None, None] / 1000


def test_frozen_epsilon_applies_vendor_cfg_and_microbatching():
    model = TinyDenoiser().requires_grad_(False).eval()
    state, _, labels = inputs()
    t = torch.full((6,), 500)
    conditional, unconditional = model(state, t, y=labels), model(state, t, y=None)
    assert torch.equal(FrozenEpsilon(model, guidance_scale=0.0, microbatch=4)(state, t, labels), conditional)
    guided = FrozenEpsilon(model, guidance_scale=1.5, microbatch=4)(state, t, labels)
    assert torch.allclose(guided, conditional + 1.5 * (conditional - unconditional))
    assert torch.equal(guided, FrozenEpsilon(model, guidance_scale=1.5, microbatch=64)(state, t, labels))
    assert not guided.requires_grad


def test_frozen_epsilon_refuses_a_trainable_or_training_model():
    with pytest.raises(ValueError):
        FrozenEpsilon(TinyDenoiser().eval(), guidance_scale=0.0, microbatch=4)
    with pytest.raises(ValueError):
        FrozenEpsilon(TinyDenoiser().requires_grad_(False).train(), guidance_scale=0.0, microbatch=4)
