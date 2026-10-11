"""Gate D: the real frozen denoiser through our sampler vs the vendor DDIM sampler.

Skipped where the pretrained checkpoint is not on disk; run it on the training host.
"""
from __future__ import annotations

import importlib
import sys

import pytest
import torch

from scaffold.backbone import FixedScaffold, FrozenEpsilon, load_frozen_score
from scaffold.sampling import ScaffoldKernel, Schedule, sample, stride_grid
from scaffold.utils import ROOT, SCAFFOLD_ROOT, load_config, rooted_path

CONFIG = load_config(SCAFFOLD_ROOT / 'config.json')
pytestmark = pytest.mark.skipif(not rooted_path(CONFIG['paths']['checkpoint']).is_file(),
                                reason='the pretrained score checkpoint is not available on this machine')


@pytest.mark.parametrize('guidance_scale', [0.0, 1.5])
def test_lambda0_matches_vendor_ddim_with_the_pretrained_score(guidance_scale):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    score, _ = load_frozen_score(CONFIG, device)
    vendor_root = ROOT / 'third_party' / 'CCUA-DDPM'
    sys.path.insert(0, str(vendor_root))
    try:
        vendor = importlib.import_module('diffusion')
    finally:
        sys.path.remove(str(vendor_root))
        sys.modules.pop('diffusion', None)
    diffusion = CONFIG['diffusion']
    reference = vendor.GaussianDiffusionSamplerDDIM(
        score, diffusion['beta_1'], diffusion['beta_T'], diffusion['steps'], omega=guidance_scale, cond=True).to(device)
    labels = torch.arange(8, device=device) * 13 % CONFIG['dataset']['num_classes']
    initial = torch.randn((8, 3, 32, 32), device=device, generator=torch.Generator(device=device).manual_seed(0))
    with torch.no_grad():
        expected = reference(initial, labels, method='ddim', skip=10)
    schedule = Schedule.from_config(CONFIG)
    ours = sample(FrozenEpsilon(score, guidance_scale=guidance_scale, microbatch=8),
                  ScaffoldKernel(schedule, FixedScaffold(0.0)), labels, initial,
                  stride_grid(len(schedule), 10), generator=torch.Generator(device=device).manual_seed(1))
    error = float((ours.clamp(-1, 1) - expected).abs().max())
    assert error < 5e-3, f'max abs pixel difference vs vendor DDIM: {error}'
