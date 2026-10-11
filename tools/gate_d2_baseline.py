"""Gate D.2: does the same checkpoint under the same protocol give the same baseline distribution and metrics?

    python tools/gate_d2_baseline.py --output scaffold/outputs/gate_d2 [--num-images 5000] [--stages ddim ddpm]

For the real pretrained score this generates paired sample sets (same labels, same initial latents, same
Brownian draws) from the pinned vendor samplers and from scaffold/sampling.py, then scores every set with
scaffold/metrics.py and with the vendor FID code. Nothing is trained and no scaffold source is modified.

    ddim   vendor DDIM (skip 50 / 20 / 10) vs lambda = 0 on the same stride grid, at CFG 0 and 1.5;
           the 'leading' 20 / 50 NFE grids (stride grid plus the clean edge); a 1e-6 input-nudge control
           that measures how far round-off-sized perturbations move the endpoint and the FID.
    ddpm   native 1000-step ancestral DDPM (fixedlarge): vendor mean / variance vs DDPMKernel, shared noise;
           and a check that the vendor's own forward() equals that loop.

Every cell is resumable: a finished sample set or metric file is reused.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scaffold.backbone import FixedScaffold, FrozenEpsilon, load_frozen_score  # noqa: E402
from scaffold.main import BROWNIAN_STREAM, LATENT_STREAM, _vendor_diffusion, to_unit_nhwc  # noqa: E402
from scaffold.metrics import compute_metrics, metric_asset_binding  # noqa: E402
from scaffold.sampling import CLEAN, DDPMKernel, ScaffoldKernel, Schedule, respaced_grid, sample, stride_grid  # noqa: E402
from scaffold.utils import SCAFFOLD_ROOT, atomic_json, atomic_numpy, load_config, rooted_path, source_hashes, stream_generator  # noqa: E402

SEED, BATCH = 0, 500


class ForwardCounter:
    """Counts images pushed through theta, so the real number of model forwards per sample is on record."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.images = 0
        model.register_forward_hook(lambda module, inputs, output: self.add(len(inputs[0])))

    def add(self, count: int) -> None:
        self.images += count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-images', type=int, default=5000)
    parser.add_argument('--stages', nargs='+', default=['ddim', 'ddpm'], choices=['ddim', 'ddpm'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)
    config = load_config(SCAFFOLD_ROOT / 'config.json')
    diffusion, classes, size = config['diffusion'], int(config['dataset']['num_classes']), int(config['dataset']['image_size'])
    model, score_sha = load_frozen_score(config, device)
    counter = ForwardCounter(model)
    schedule = Schedule.from_config(config)
    vendor = _vendor_diffusion()
    labels_all = np.arange(args.num_images, dtype=np.int64) % classes
    metrics_root = rooted_path(config['evaluation']['metrics_root'])
    metric_asset_binding(metrics_root)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / 'labels.npy', labels_all)

    def batches():
        for batch, start in enumerate(range(0, args.num_images, BATCH)):
            labels = torch.as_tensor(labels_all[start:start + BATCH], device=device)
            initial = torch.randn((len(labels), 3, size, size), device=device,
                                  generator=stream_generator(device, SEED, LATENT_STREAM, batch))
            yield batch, labels, initial

    def generate(name: str, function) -> None:
        """function(batch, labels, initial) -> endpoint in [-1, 1] (unclamped is fine)."""
        path = args.output / f'{name}.npy'
        if path.is_file():
            return
        counter.images, started, parts = 0, time.perf_counter(), []
        with torch.no_grad():
            for batch, labels, initial in batches():
                parts.append(to_unit_nhwc(function(batch, labels, initial)))
        atomic_numpy(path, np.concatenate(parts))
        atomic_json(args.output / f'{name}.generation.json', {
            'model_forwards_per_image': counter.images / args.num_images, 'seconds': time.perf_counter() - started})
        print(f'generated {name}: {counter.images / args.num_images:.1f} forwards/image', flush=True)

    def ours(kernel, grid, scale, nudge=0.0):
        epsilon = FrozenEpsilon(model, guidance_scale=scale, microbatch=BATCH)

        def run(batch, labels, initial):
            if nudge:
                initial = initial + nudge * torch.randn(initial.shape, device=device,
                                                        generator=stream_generator(device, SEED, 7, batch))
            return sample(epsilon, kernel, labels, initial, grid,
                          generator=stream_generator(device, SEED, BROWNIAN_STREAM, batch))
        return run

    def vendor_ddim(stride, scale):
        sampler = vendor.GaussianDiffusionSamplerDDIM(model, diffusion['beta_1'], diffusion['beta_T'], diffusion['steps'],
                                                      omega=scale, cond=True).to(device)
        return lambda batch, labels, initial: sampler(initial, labels, method='ddim', skip=stride)

    def vendor_ddpm(variance_type):
        sampler = vendor.GaussianDiffusionSampler(model, diffusion['beta_1'], diffusion['beta_T'], diffusion['steps'],
                                                  num_class=classes, var_type=variance_type).to(device)

        def run(batch, labels, initial):
            generator = stream_generator(device, SEED, BROWNIAN_STREAM, batch)
            state = initial.clone()
            for step in range(diffusion['steps'] - 1, -1, -1):
                mean, log_variance = sampler.p_mean_variance(state, torch.full_like(labels, step), y=labels, omega=0.0)
                noise = torch.randn(state.shape, device=device, generator=generator) if step > 0 else 0
                state = mean + torch.exp(0.5 * log_variance) * noise
            return state
        return run

    ddim_kernel = ScaffoldKernel(schedule, FixedScaffold(0.0))
    pairs: list[tuple[str, str, str]] = []  # (label, reference set, candidate set)
    if 'ddim' in args.stages:
        for scale in (0.0, 1.5):
            for stride in (50, 20, 10):
                tag = f's{stride}_w{scale}'
                generate(f'vendor_ddim_{tag}', vendor_ddim(stride, scale))
                generate(f'ours_ddim_{tag}', ours(ddim_kernel, stride_grid(len(schedule), stride), scale))
                pairs.append((f'ddim stride {stride} cfg {scale}: vendor vs ours', f'vendor_ddim_{tag}', f'ours_ddim_{tag}'))
            for nfe in (20, 50):
                generate(f'ours_ddim_leading{nfe}_w{scale}', ours(ddim_kernel, respaced_grid(len(schedule), nfe), scale))
                pairs.append((f'ddim cfg {scale}: stride grid vs leading {nfe} NFE (adds the clean edge)',
                              f'ours_ddim_s{1000 // nfe}_w{scale}', f'ours_ddim_leading{nfe}_w{scale}'))
            generate(f'ours_ddim_s10_w{scale}_nudge', ours(ddim_kernel, stride_grid(len(schedule), 10), scale, nudge=1e-6))
            pairs.append((f'ddim stride 10 cfg {scale}: ours vs ours with a 1e-6 input nudge',
                          f'ours_ddim_s10_w{scale}', f'ours_ddim_s10_w{scale}_nudge'))
    forward_check = None
    if 'ddpm' in args.stages:
        full = respaced_grid(len(schedule), len(schedule))
        generate('vendor_ddpm1000_fixedlarge', vendor_ddpm('fixedlarge'))
        generate('ours_ddpm1000_fixedlarge', ours(DDPMKernel(schedule, variance_type='fixedlarge'), full, 0.0))
        pairs.append(('ddpm 1000 fixedlarge cfg 0: vendor vs ours', 'vendor_ddpm1000_fixedlarge', 'ours_ddpm1000_fixedlarge'))
        # The vendor's own forward() (random labels, global RNG) against the controlled loop used above.
        sampler = vendor.GaussianDiffusionSampler(model, diffusion['beta_1'], diffusion['beta_T'], diffusion['steps'],
                                                  num_class=classes, var_type='fixedlarge').to(device)
        initial = torch.randn((16, 3, size, size), device=device, generator=torch.Generator(device).manual_seed(11))
        torch.manual_seed(123)
        native, native_labels = sampler(initial, omega=0.0)
        torch.manual_seed(123)
        labels = torch.randint(0, classes, (16,)).to(device)
        state = initial.clone()
        with torch.no_grad():
            for step in range(diffusion['steps'] - 1, -1, -1):
                mean, log_variance = sampler.p_mean_variance(state, torch.full_like(labels, step), y=labels, omega=0.0)
                noise = torch.randn_like(state) if step > 0 else 0
                state = mean + torch.exp(0.5 * log_variance) * noise
        forward_check = {'labels_equal': bool(torch.equal(labels, native_labels)),
                         'max_abs_difference': float((state.clamp(-1, 1) - native).abs().max())}

    # Metrics: scaffold/metrics.py on every set, and the vendor FID code on the same files.
    sys.path.insert(0, str(ROOT / 'third_party' / 'CCUA-DDPM'))
    vendor_fid = importlib.import_module('score.fid')
    manifest = json.loads((metrics_root / 'cifar100.metric_manifest.json').read_text())
    settings = {**config['evaluation'], 'recall': True}
    results = {}
    for path in sorted(args.output.glob('*.npy')):
        if path.name == 'labels.npy' or path.name.endswith('.inception.npy'):
            continue
        report = args.output / f'{path.stem}.metrics.json'
        if not report.is_file():
            images = np.load(path, mmap_mode='r')
            ours_metrics = compute_metrics(path, labels_all, settings, metrics_root, device)
            again = compute_metrics(path, labels_all, {**settings, 'recall': False}, metrics_root, device)
            vendor_value = float(vendor_fid.get_fid_score(str(metrics_root / manifest['fid_file']),
                                                         (image.transpose(2, 0, 1) for image in images),
                                                         num_images=len(images), batch_size=50))
            atomic_json(report, {'scaffold': ours_metrics, 'vendor_fid_overall': vendor_value,
                                 'scaffold_recomputed_fid_overall': again['FID']['Overall']})
            print(f'scored {path.stem}: FID {ours_metrics["FID"]["Overall"]:.4f} (vendor code {vendor_value:.4f})', flush=True)
        results[path.stem] = {**json.loads(report.read_text()),
                              **json.loads((args.output / f'{path.stem}.generation.json').read_text())}

    comparisons = []
    for label, reference, candidate in pairs:
        a, b = np.load(args.output / f'{reference}.npy'), np.load(args.output / f'{candidate}.npy')
        difference = np.abs(a - b) * 2  # back to the [-1, 1] pixel scale used by the samplers
        fid_a, fid_b = (results[name]['scaffold']['FID'] for name in (reference, candidate))
        comparisons.append({
            'comparison': label, 'reference': reference, 'candidate': candidate,
            'max_abs_pixel_difference': float(difference.max()), 'mean_abs_pixel_difference': float(difference.mean()),
            'fid_overall': [fid_a['Overall'], fid_b['Overall']], 'delta_fid_overall': fid_b['Overall'] - fid_a['Overall'],
            'delta_fid_few': fid_b['Few'] - fid_a['Few'],
            'delta_is_overall': results[candidate]['scaffold']['IS']['Overall']['mean'] - results[reference]['scaffold']['IS']['Overall']['mean'],
            'delta_recall': results[candidate]['scaffold']['Recall'] - results[reference]['scaffold']['Recall']})
    metric_agreement = {name: {'scaffold_fid': entry['scaffold']['FID']['Overall'], 'vendor_fid': entry['vendor_fid_overall'],
                               'difference': entry['scaffold']['FID']['Overall'] - entry['vendor_fid_overall'],
                               'recompute_difference': entry['scaffold']['FID']['Overall'] - entry['scaffold_recomputed_fid_overall']}
                        for name, entry in results.items()}
    atomic_json(args.output / 'gate_d2_report.json', {
        'score_sha256': score_sha, 'source_sha256': source_hashes(), 'num_images': args.num_images, 'seed': SEED,
        'labels': 'cyclic class-balanced', 'batch_size': BATCH, 'sets': results, 'comparisons': comparisons,
        'metric_agreement': metric_agreement, 'vendor_forward_vs_loop': forward_check})
    print('report written', flush=True)


if __name__ == '__main__':
    main()
