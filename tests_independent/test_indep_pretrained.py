"""Positive controls that need the pinned pretrained score. They skip when the checkpoint is absent."""
from pathlib import Path

import numpy as np
import pytest
import torch

from scaffold import utils as U
from scaffold.backbone import FrozenEpsilon, load_frozen_score
from scaffold.sampling import Schedule

REPOSITORY = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.real_assets


@pytest.fixture(scope='module')
def problem():
    config = U.load_config(REPOSITORY / 'scaffold' / 'config.json')
    if not U.rooted_path(config['paths']['checkpoint']).is_file():
        pytest.skip('the pretrained checkpoint is not on this machine')
    if not (REPOSITORY / 'data' / 'cifar-100-python').is_dir():
        pytest.skip('data/cifar-100-python is absent')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, digest = load_frozen_score(config, device)
    assert digest == config['paths']['checkpoint_sha256']
    images, indices, _ = U.load_training_data(config)
    chosen = np.concatenate([class_indices[:2] for class_indices in indices[::4]])
    labels = torch.as_tensor(np.repeat(np.arange(100)[::4], 2), device=device)
    return config, FrozenEpsilon(model, guidance_scale=0.0, microbatch=16), U.to_pixels(images[chosen], device), labels


@pytest.mark.parametrize('index', [50, 300, 600, 950])
def test_frozen_score_predicts_the_forward_noise_at_the_right_timestep(problem, index):
    """Checks the timestep convention end to end: index i is noise level alpha_bar[i].

    Convention: the denoiser must be worse when told a timestep 150 indices away, in each direction
    that stays inside the schedule.
    Quality: up to index 600 it must beat the trivial predictor eps = x_t sqrt(1 - ab), which is exact
    only for N(0, I) data. At index 950 that predictor is near-optimal (error O(alpha_bar) ~ 3e-5), so
    no network can be required to beat it; the denoiser must stay within a factor 10 of it instead.
    Measured with the pinned checkpoint on 2026-10-11: mse 0.103 / 0.0116 / 0.0027 / 4.0e-5 at indices
    50 / 300 / 600 / 950, against 0.455 / 0.042 / 0.0046 / 0.40 at a wrong timestep.
    """
    config, epsilon, x0, labels = problem
    alpha = float(Schedule.from_config(config).alpha_bar[index])
    noise = torch.randn(x0.shape, device=x0.device, generator=torch.Generator(x0.device).manual_seed(index))
    x_t = alpha ** 0.5 * x0 + (1 - alpha) ** 0.5 * noise
    t = torch.full_like(labels, index)
    error = float((epsilon(x_t, t, labels) - noise).square().mean())
    trivial = float((x_t * (1 - alpha) ** 0.5 - noise).square().mean())
    shifted = [float((epsilon(x_t, torch.full_like(labels, told), labels) - noise).square().mean())
               for told in (index - 150, index + 150) if 0 <= told < 1000]
    print(f't={index}: mse {error:.2e}, trivial {trivial:.2e}, wrong timestep {shifted}')
    assert all(error < value for value in shifted)
    assert error < trivial if index <= 600 else error < 10 * trivial
