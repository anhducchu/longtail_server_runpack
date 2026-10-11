"""backbone.py: scaffold contracts, controls, CFG wrapper and the frozen-score guard."""
import hashlib
import math
from importlib import import_module

import pytest
import torch
from torch import nn

from scaffold import backbone as B


def batch(count=6, size=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(count, 3, size, size, generator=generator), torch.rand(count, generator=generator) * 0.9 + 0.05,
            torch.randint(0, 10, (count,), generator=generator))


def scaffolds(size=8, **head):
    return {'mlp': B.MLPScaffold(10, image_size=size, hidden_dims=(32, 16), condition_dim=16, **head),
            'unet': B.UNetScaffold(10, base_channels=8, channel_multipliers=(1, 2, 4), residual_blocks=1,
                                   condition_dim=16, **head)}


def randomise_output(scaffold, seed=1):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        scaffold.output.weight.copy_(torch.randn(scaffold.output.weight.shape, generator=generator) * 0.3)
        scaffold.output.bias.copy_(torch.randn(scaffold.output.bias.shape, generator=generator))
    return scaffold.eval()


@pytest.mark.parametrize('kind', ['mlp', 'unet'])
@pytest.mark.parametrize('initial', [1.0, 0.25, 3.0])
def test_learned_scaffolds_start_exactly_at_variance_initial(kind, initial):
    state, time, labels = batch()
    lam = scaffolds(variance_initial=initial)[kind](state, time, labels)
    assert lam.shape == state.shape
    torch.testing.assert_close(lam, torch.full_like(state, initial), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('kind', ['mlp', 'unet'])
def test_learned_scaffolds_are_positive_finite_and_state_time_class_dependent(kind):
    scaffold = randomise_output(scaffolds(variance_floor=1e-3)[kind])
    state, time, labels = batch()
    lam = scaffold(state, time, labels).detach()
    assert lam.shape == state.shape and torch.isfinite(lam).all() and float(lam.min()) >= 1e-3
    extreme = scaffold(state * 1e4, time, labels).detach()
    assert torch.isfinite(extreme).all() and float(extreme.min()) >= 1e-3
    assert not torch.allclose(lam, scaffold(state + 0.5, time, labels))
    assert not torch.allclose(lam, scaffold(state, 1.0 - time, labels))
    assert not torch.allclose(lam, scaffold(state, time, (labels + 1) % 10))
    assert float(lam.std()) > 0, 'lambda should vary across pixels'


@pytest.mark.parametrize('kind', ['mlp', 'unet'])
def test_scaffold_output_for_a_sample_does_not_depend_on_its_batch(kind):
    scaffold = randomise_output(scaffolds()[kind])
    state, time, labels = batch()
    together = scaffold(state, time, labels)
    alone = torch.cat([scaffold(state[i:i + 1], time[i:i + 1], labels[i:i + 1]) for i in range(len(state))])
    torch.testing.assert_close(together, alone, rtol=1e-4, atol=1e-5)
    scalar_time = scaffold(state, 0.37, labels)
    torch.testing.assert_close(scalar_time, scaffold(state, torch.full((len(state),), 0.37), labels), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('kind', ['mlp', 'unet'])
def test_every_parameter_can_receive_gradient(kind):
    scaffold = randomise_output(scaffolds()[kind]).train()
    state, time, labels = batch()
    scaffold(state, time, labels).log().sum().backward()
    dead = [name for name, parameter in scaffold.named_parameters()
            if parameter.grad is None or float(parameter.grad.abs().max()) == 0.0]
    used = set(labels.tolist())
    assert all(name == 'condition.label.weight' for name in dead) or not dead, dead
    label_gradient = scaffold.condition.label.weight.grad.abs().sum(dim=1) > 0
    assert set(torch.nonzero(label_gradient).flatten().tolist()) == used


def test_unet_scaffold_handles_sizes_that_do_not_halve_evenly():
    scaffold = randomise_output(scaffolds()['unet'])
    for size in (32, 10, 7):
        state, time, labels = batch(count=2, size=size)
        assert scaffold(state, time, labels).shape == state.shape


@pytest.mark.parametrize('floor,initial', [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (1e-4, float('inf'))])
def test_positive_head_rejects_inconsistent_floor_and_initial(floor, initial):
    with pytest.raises(ValueError):
        B.MLPScaffold(10, image_size=8, variance_floor=floor, variance_initial=initial)


def test_positive_head_formula():
    head = B._PositiveHead(0.01, 0.7)
    raw = torch.linspace(-30, 30, 13)
    expected = torch.tensor([0.01 + math.log1p(math.exp(value + math.log(math.expm1(0.69)))) for value in raw.tolist()])
    torch.testing.assert_close(head(raw), expected, rtol=1e-5, atol=1e-7)
    assert float(head(torch.tensor(-1e4))) == pytest.approx(0.01)


def test_architecture_argument_validation():
    with pytest.raises(ValueError):
        B.MLPScaffold(10, hidden_dims=())
    with pytest.raises(ValueError):
        B.MLPScaffold(10, hidden_dims=(16, 0))
    for arguments in ({'base_channels': 12}, {'channel_multipliers': ()}, {'residual_blocks': 0}):
        with pytest.raises(ValueError):
            B.UNetScaffold(10, **arguments)


# Controls

def test_fixed_scaffold_is_a_broadcastable_constant():
    state, time, labels = batch()
    lam = B.FixedScaffold(0.25)(state.double(), time, labels)
    assert lam.shape == (6, 1, 1, 1) and lam.dtype == torch.float64 and (lam == 0.25).all()
    assert not list(B.FixedScaffold(0.25).parameters())
    for value in (-0.1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            B.FixedScaffold(value)


@pytest.mark.parametrize('kind', ['mlp', 'unet'])
def test_time_only_wrapper_hides_the_state(kind):
    inner = randomise_output(scaffolds()[kind])
    wrapped = B.TimeOnlyScaffold(inner)
    state, time, labels = batch()
    lam = wrapped(state, time, labels)
    assert torch.equal(lam, wrapped(torch.randn_like(state) * 9, time, labels))
    assert torch.equal(lam, inner(torch.zeros_like(state), time, labels))
    assert not torch.allclose(lam, wrapped(state, 1.0 - time, labels))
    assert not torch.allclose(lam, wrapped(state, time, (labels + 1) % 10))


def test_shuffled_wrapper_borrows_another_state_of_the_same_class_only():
    inner = randomise_output(scaffolds()['mlp'])
    wrapped = B.ShuffledStateScaffold(inner)
    state, time, _ = batch(count=7)
    labels = torch.tensor([3, 5, 3, 5, 3, 5, 5])
    lam = wrapped(state, time, labels)
    donors = []
    for i in range(len(state)):
        same = [j for j in range(len(state)) if labels[j] == labels[i]]
        donor = same[same.index(i) - 1]  # previous member of the class, cyclically
        donors.append(donor)
        expected = inner(state[donor:donor + 1], time[i:i + 1], labels[i:i + 1])
        torch.testing.assert_close(lam[i:i + 1], expected, rtol=1e-4, atol=1e-5)
    assert all(donor != i and labels[donor] == labels[i] for i, donor in enumerate(donors))
    assert sorted(donors) == list(range(7)), 'a permutation: every state is used exactly once'
    assert not torch.allclose(lam, inner(state, time, labels))
    with pytest.raises(ValueError, match='two samples'):
        wrapped(state, time, torch.tensor([3, 5, 3, 5, 3, 5, 7]))


def settings(**changes):
    base = {'kind': 'mlp', 'inputs': 'state_time_class', 'image_size': 8, 'hidden_dims': [16], 'condition_dim': 8,
            'variance_floor': 1e-4, 'variance_initial': 1.0, 'base_channels': 8, 'channel_multipliers': [1, 2],
            'residual_blocks': 1}
    return {**base, **changes}


def test_build_scaffold_dispatch():
    assert isinstance(B.build_scaffold(settings(), 10), B.MLPScaffold)
    assert isinstance(B.build_scaffold(settings(kind='unet'), 10), B.UNetScaffold)
    time_only = B.build_scaffold(settings(inputs='time_class'), 10)
    assert isinstance(time_only, B.TimeOnlyScaffold) and isinstance(time_only.scaffold, B.MLPScaffold)
    assert B.build_scaffold(settings(), 7).condition.label.num_embeddings == 7
    for changes in ({'kind': 'cnn'}, {'inputs': 'state'}):
        with pytest.raises(ValueError):
            B.build_scaffold(settings(**changes), 10)


def test_time_only_and_full_scaffolds_share_one_parameterisation():
    """The time-only control must differ from the learned arm by its input only, not by capacity."""
    count = lambda module: sum(parameter.numel() for parameter in module.parameters())
    assert count(B.build_scaffold(settings(), 10)) == count(B.build_scaffold(settings(inputs='time_class'), 10))


# Frozen denoiser

class FakeTheta(nn.Module):
    """Per-sample model with distinct conditional / unconditional outputs; counts forward passes."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.calls = []

    def forward(self, x, t, y=None):
        self.calls.append((len(x), y is None))
        shift = t.to(x).reshape(-1, 1, 1, 1) * 0.01
        if y is None:
            return self.weight * x - shift
        return self.weight * x + shift + y.to(x).reshape(-1, 1, 1, 1)


def frozen(model=None):
    return (model or FakeTheta()).requires_grad_(False).eval()


@pytest.mark.parametrize('scale', [0.0, 0.8, 3.0])
@pytest.mark.parametrize('microbatch', [1, 4, 100])
def test_frozen_epsilon_implements_vendor_cfg_and_microbatching(scale, microbatch):
    x, _, labels = batch(count=10)
    t = torch.arange(10) * 7
    model = frozen()
    got = B.FrozenEpsilon(model, guidance_scale=scale, microbatch=microbatch)(x, t, labels)
    conditional, unconditional = model(x, t, y=labels), model(x, t, y=None)
    torch.testing.assert_close(got, (1 + scale) * conditional - scale * unconditional, rtol=1e-5, atol=1e-5)
    assert got.shape == x.shape and not got.requires_grad


def test_frozen_epsilon_costs_one_pass_without_cfg_and_two_with():
    x, _, labels = batch(count=10)
    t = torch.zeros(10, dtype=torch.long)
    plain, guided = frozen(), frozen()
    B.FrozenEpsilon(plain, guidance_scale=0.0, microbatch=4)(x, t, labels)
    B.FrozenEpsilon(guided, guidance_scale=1.0, microbatch=4)(x, t, labels)
    assert plain.calls == [(4, False), (4, False), (2, False)]
    assert guided.calls == [(4, False), (4, True)] * 2 + [(2, False), (2, True)]


def test_frozen_epsilon_refuses_trainable_or_training_models():
    with pytest.raises(ValueError):
        B.FrozenEpsilon(FakeTheta().eval(), guidance_scale=0.0, microbatch=4)
    with pytest.raises(ValueError):
        B.FrozenEpsilon(FakeTheta().requires_grad_(False).train(), guidance_scale=0.0, microbatch=4)
    for scale, microbatch in ((float('nan'), 4), (float('inf'), 4), (0.0, 0)):
        with pytest.raises(ValueError):
            B.FrozenEpsilon(frozen(), guidance_scale=scale, microbatch=microbatch)


def test_frozen_epsilon_drives_the_vendor_unet_interface():
    """The real architecture (random weights): conditional and unconditional calls must both work."""
    unet = import_module('third_party.CCUA-DDPM.model.model').UNet
    torch.manual_seed(0)
    model = unet(T=50, ch=32, ch_mult=[1, 2], attn=[1], num_res_blocks=1, dropout=0.1, cond=True, num_class=10)
    model.requires_grad_(False).eval()
    x, _, labels = batch(count=3, size=16)
    t = torch.tensor([0, 25, 49])
    got = B.FrozenEpsilon(model, guidance_scale=1.5, microbatch=2)(x, t, labels)
    expected = 2.5 * model(x, t, y=labels) - 1.5 * model(x, t, y=None)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)
    plain = B.FrozenEpsilon(model, guidance_scale=0.0, microbatch=2)(x, t, labels)
    torch.testing.assert_close(plain, model(x, t, y=labels), rtol=1e-4, atol=1e-5)


def test_load_frozen_score_refuses_a_file_with_the_wrong_hash(tmp_path):
    path = tmp_path / 'score.pt'
    path.write_bytes(b'not the pinned checkpoint')
    config = {'paths': {'checkpoint': str(path), 'checkpoint_sha256': hashlib.sha256(b'pinned').hexdigest()}}
    with pytest.raises(RuntimeError, match='SHA mismatch'):
        B.load_frozen_score(config, torch.device('cpu'))
