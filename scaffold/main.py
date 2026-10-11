"""Train the scaffold phi and evaluate samplers, for one frozen denoiser theta.

    python -m scaffold.main preflight                      Gate D: real-checkpoint baseline checks
    python -m scaffold.main diagnose  [--checkpoint PATH]  edge-level loss and gradient diagnostics
    python -m scaffold.main train     [--resume]
    python -m scaffold.main evaluate  [--arms NAME ...]

Every command takes --config (default scaffold/config.json), --device and --set key=value.

Training draws class-uniform batches, so the objective is class-balanced: the
rarity weight N / (C n_c) and the proposal correction C n_c / N cancel exactly.

Evaluation samples every (seed, arm) cell under one shared protocol (score,
grid, CFG, sample count):
    * noise comes from SeedSequence(seed, stream, batch): seeds never share a
      batch, and all arms of a seed consume the same latents and Brownian draws;
    * each batch is published atomically as its own shard, and resuming skips
      only shards present on disk, so a lost file is regenerated, never zero-filled;
    * a learned arm contributes only its checkpoint. The ``evaluation`` block is
      not part of a checkpoint's identity, but a checkpoint trained for another
      score, kernel, grid or CFG is refused unless the arm says so explicitly.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from scaffold import __version__
from scaffold.backbone import FixedScaffold, FrozenEpsilon, ShuffledStateScaffold, build_scaffold, load_frozen_score
from scaffold.loss import RespacedVLB, build_loss, forward_sample
from scaffold.metrics import GROUPS, MANIFEST, compute_metrics, inception_statistics, metric_asset_binding
from scaffold.sampling import (
    KERNEL_VERSION, CLEAN, DDPMKernel, EpsilonFn, Grid, ScaffoldKernel, Schedule, Transition, build_grid,
    posterior_variance, sample, stride_grid,
)
from scaffold.utils import (
    ROOT, SCAFFOLD_ROOT, array_sha256, atomic_json, atomic_numpy, atomic_torch_save, canonical_hash, load_config,
    load_heldout_data, load_training_data, output_directory, rooted_path, set_reproducible_seed, sha256_file,
    source_hashes, stream_generator, to_pixels,
)

LOGGER = logging.getLogger(__name__)
TRAINING_SCHEMA = 'ddim_scaffold_training_v1'
EVALUATION_SCHEMA = 'ddim_scaffold_evaluation_v1'
LATENT_STREAM, BROWNIAN_STREAM = 0, 1
# Preflight tolerances; each is justified where it is used.
STEP_TOLERANCE = 1e-5          # one edge, [-1, 1] pixel units: float32 round-off is ~1e-6, a wrong formula >= 1e-3
ENDPOINT_AMPLIFICATION = 10.0  # endpoint gap allowed relative to the measured effect of a 1e-6 input nudge
ENDPOINT_FLOOR = 1e-4          # absolute endpoint gap always accepted (1 / 80 of a grey level)
NEAR_NOISE_FACTOR = 10.0       # near pure noise theta must be within this factor of the analytic predictor
Dataset = tuple[np.ndarray, list[np.ndarray], np.ndarray]
Panel = tuple[np.ndarray, np.ndarray]
MetricFn = Callable[[Path, np.ndarray], dict[str, Any]]


def frozen_epsilon(config: dict[str, Any], device: torch.device) -> tuple[EpsilonFn, str]:
    """The guided frozen denoiser of this config and the SHA-256 of its checkpoint."""
    score, digest = load_frozen_score(config, device)
    return FrozenEpsilon(score, guidance_scale=float(config['sampler']['guidance_scale']),
                         microbatch=int(config['model']['microbatch'])), digest


# Training

def training_identity(config: dict[str, Any]) -> dict[str, Any]:
    """The part of a config that defines a trained scaffold: everything but ``evaluation``."""
    return {name: value for name, value in config.items() if name != 'evaluation'}


def draw_batch(images: np.ndarray, indices: list[np.ndarray], settings: dict[str, Any],
               rng: np.random.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform classes, then uniform images within each class, with optional horizontal flip."""
    per_class = int(settings['samples_per_class'])
    classes = rng.choice(len(indices), size=int(settings['classes_per_batch']), replace=False)
    chosen = np.concatenate([rng.choice(indices[c], size=per_class, replace=len(indices[c]) < per_class)
                             for c in classes])
    clean = to_pixels(images[chosen], device)
    if settings['horizontal_flip']:
        flip = torch.as_tensor(rng.random(len(clean)) < 0.5, device=device)
        clean[flip] = clean[flip].flip(-1)
    return clean, torch.as_tensor(np.repeat(classes, per_class), device=device, dtype=torch.long)


def validation_panels(config: dict[str, Any], images: np.ndarray, indices: list[np.ndarray],
                      heldout: Panel | None) -> dict[str, Panel]:
    """Fixed image panels: long-tailed training images and, if given, held-out test images."""
    per_class = int(config['validation']['per_class'])
    rng = np.random.default_rng(int(config['validation']['seed']))
    chosen = [rng.permutation(class_indices)[:per_class] for class_indices in indices]
    labels = np.concatenate([np.full(len(ids), class_id) for class_id, ids in enumerate(chosen)])
    panels = {'train': (images[np.concatenate(chosen)], labels)}
    if heldout is not None:
        heldout_images, heldout_labels = heldout
        chosen = np.concatenate([rng.permutation(np.flatnonzero(heldout_labels == class_id))[:per_class]
                                 for class_id in range(len(indices))])
        panels['heldout'] = (heldout_images[chosen], heldout_labels[chosen])
    return panels


@torch.no_grad()
def panel_bound(loss: RespacedVLB, epsilon_fn: EpsilonFn, kernel: nn.Module, panel: Panel, *,
                seed: int, batch_size: int, device: torch.device) -> dict[str, float]:
    """Bound on a fixed panel, in nats per dimension, overall and per frequency group.

    Each call reuses the same (image, edge, noise) triples, so values are
    comparable across checkpoints and across scaffolds.
    """
    images, labels = panel
    generator = torch.Generator(device=device).manual_seed(seed)
    edges = torch.randperm(len(images), generator=generator, device=device) % loss.num_edges
    values = []
    for start in range(0, len(images), batch_size):
        chunk = slice(start, start + batch_size)
        x0 = to_pixels(images[chunk], device)
        y = torch.as_tensor(labels[chunk], device=device, dtype=torch.long)
        noise = torch.randn(x0.shape, device=device, generator=generator)
        kl = loss.edge_kl(epsilon_fn, kernel, x0, y, edges[chunk], noise)['kl']
        values.append(loss.num_edges * kl / x0[0].numel())
    per_sample = torch.cat(values).cpu().numpy()
    result = {'Overall': float(per_sample.mean())}
    for name, classes in GROUPS.items():
        members = np.isin(labels, classes)
        if members.any():
            result[name] = float(per_sample[members].mean())
    return result


def validate(loss: RespacedVLB, epsilon_fn: EpsilonFn, schedule: Schedule, scaffold: nn.Module,
             panels: dict[str, Panel], config: dict[str, Any], device: torch.device) -> dict[str, dict[str, float]]:
    was_training = scaffold.training
    scaffold.eval()
    try:
        kernel = ScaffoldKernel(schedule, scaffold)
        return {name: panel_bound(loss, epsilon_fn, kernel, panel, seed=int(config['validation']['seed']),
                                  batch_size=int(config['validation']['batch_size']), device=device)
                for name, panel in panels.items()}
    finally:
        scaffold.train(was_training)


def training_step(loss: RespacedVLB, epsilon_fn: EpsilonFn, kernel: ScaffoldKernel,
                  optimizer: torch.optim.Optimizer, clean: torch.Tensor, labels: torch.Tensor,
                  generator: torch.Generator, *, gradient_clip: float) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    value, details = loss(epsilon_fn, kernel, clean, labels, generator=generator)
    value.backward()
    norm = torch.nn.utils.clip_grad_norm_(kernel.scaffold.parameters(), gradient_clip)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('nonfinite phi gradient')
    optimizer.step()
    multiplier = details['multiplier']
    row = {'loss_nats_per_dim': float(value.detach()), 'gradient_norm': float(norm),
           'multiplier_mean': float(multiplier.mean()), 'multiplier_min': float(multiplier.min()),
           'multiplier_max': float(multiplier.max()),
           'capped_fraction': float(details['capped'].float().mean())}
    if 'requested_multiplier' in details:
        row['requested_multiplier_mean'] = float(details['requested_multiplier'].mean())
    return row


@torch.no_grad()
def update_ema(average: nn.Module, online: nn.Module, decay: float) -> None:
    for target, source in zip(average.parameters(), online.parameters()):
        target.lerp_(source, 1.0 - decay)


def load_scaffold(path: Path, device: torch.device, *, weights: str = 'ema') -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a trained scaffold from its checkpoint alone; the training config travels inside it."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('schema') != TRAINING_SCHEMA:
        raise RuntimeError(f'{path} is not a {TRAINING_SCHEMA} checkpoint')
    if weights not in ('ema', 'online'):
        raise ValueError("weights must be 'ema' or 'online'")
    state = payload['ema'] if weights == 'ema' else payload['scaffold']
    if state is None:
        raise RuntimeError("checkpoint has no EMA weights; use weights='online'")
    config = payload['config']
    scaffold = build_scaffold(config['scaffold'], int(config['dataset']['num_classes'])).to(device)
    scaffold.load_state_dict(state, strict=True)
    return scaffold.requires_grad_(False).eval(), payload


def train(config: dict[str, Any], device: torch.device, *, resume: bool = False, output: Path | None = None,
          epsilon: tuple[EpsilonFn, str] | None = None, data: Dataset | None = None,
          heldout: Panel | None = None) -> Path:
    """Run or resume training; returns the run directory.

    ``output``, ``epsilon`` (function, score SHA), ``data`` and ``heldout``
    replace the configured locations and assets; tests use them to run on CPU.
    """
    settings = config['training']
    output = output_directory(str(settings['name'])) if output is None else Path(output)
    latest = output / 'latest.pt'
    output.mkdir(parents=True, exist_ok=True)
    if resume and not latest.is_file():
        raise FileNotFoundError('--resume needs latest.pt')
    if not resume and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; pass --resume or choose another training.name')

    seed = int(settings['seed'])
    set_reproducible_seed(seed)
    epsilon_fn, score_sha = frozen_epsilon(config, device) if epsilon is None else epsilon
    schedule = Schedule.from_config(config)
    grid = build_grid(len(schedule), config['sampler']['grid'])
    loss = build_loss(config['loss'], schedule, grid)
    if getattr(loss, 'CAVEAT', None):
        LOGGER.warning('objective caveat: %s', loss.CAVEAT)
    images, indices, counts = load_training_data(config) if data is None else data
    if heldout is None and config['validation']['heldout']:
        heldout = load_heldout_data(config)
    panels = validation_panels(config, images, indices, heldout)
    identity = training_identity(config)
    provenance = {'schema': TRAINING_SCHEMA, 'kernel_version': KERNEL_VERSION,
                  'config_sha256': canonical_hash(identity), 'score_sha256': score_sha, 'grid': list(grid),
                  'images_sha256': array_sha256(images), 'class_counts': counts.astype(int).tolist()}

    scaffold = build_scaffold(config['scaffold'], len(counts)).to(device)
    kernel = ScaffoldKernel(schedule, scaffold)
    decay = float(settings['ema_decay'])
    average = copy.deepcopy(scaffold).requires_grad_(False) if decay else None
    optimizer = torch.optim.AdamW(scaffold.parameters(), lr=float(settings['learning_rate']),
                                  weight_decay=float(settings['weight_decay']))
    rng = np.random.default_rng(seed + 1)
    generator = torch.Generator(device=device).manual_seed(seed + 2)
    update, rows = 0, []
    if resume:
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved.get('provenance') != provenance:
            raise RuntimeError('latest.pt was produced by a different config, score, data or kernel')
        scaffold.load_state_dict(saved['scaffold'])
        if average is not None:
            average.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']
        generator.set_state(saved['noise_rng'])
        update, rows = int(saved['update']), list(saved['rows'])
    else:
        atomic_json(output / 'config.json', identity)
        # Matched reference for the loss: lambda = 1 (posterior variance) on the same panels.
        reference = validate(loss, epsilon_fn, schedule, FixedScaffold(1.0), panels, config, device)
        atomic_json(output / 'reference_lambda1.json', reference)
        LOGGER.info('lambda=1 reference bound: %s', json.dumps(reference))
    (output / 'train.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    total = int(settings['updates'])
    started = time.perf_counter()
    for update in range(update + 1, total + 1):
        clean, labels = draw_batch(images, indices, settings, rng, device)
        row = {'update': update, **training_step(loss, epsilon_fn, kernel, optimizer, clean, labels, generator,
                                                 gradient_clip=float(settings['gradient_clip']))}
        if average is not None:
            update_ema(average, scaffold, decay)
        checkpoint_due = update % int(settings['checkpoint_interval']) == 0 or update == total
        if checkpoint_due:
            row['validation'] = validate(loss, epsilon_fn, schedule, scaffold if average is None else average,
                                         panels, config, device)
        rows.append(row)
        with (output / 'train.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        if update == 1 or update % int(settings['log_interval']) == 0:
            LOGGER.info('update=%d/%d loss=%.6g multiplier=%.4f[%.4f,%.4f] capped=%.4f gradient=%.4g elapsed=%.0fs',
                        update, total, row['loss_nats_per_dim'], row['multiplier_mean'], row['multiplier_min'],
                        row['multiplier_max'], row['capped_fraction'], row['gradient_norm'],
                        time.perf_counter() - started)
        if checkpoint_due:
            payload = {'schema': TRAINING_SCHEMA, 'update': update,
                       'scaffold': copy.deepcopy(scaffold.state_dict()),
                       'ema': None if average is None else copy.deepcopy(average.state_dict()),
                       'optimizer': copy.deepcopy(optimizer.state_dict()),
                       'numpy_rng': copy.deepcopy(rng.bit_generator.state), 'noise_rng': generator.get_state(),
                       'config': identity, 'provenance': provenance, 'rows': rows,
                       'version': __version__, 'source_sha256': source_hashes()}
            atomic_torch_save(latest, payload)
            if settings['retain_snapshots']:
                atomic_torch_save(output / f'snapshot_{update}.pt', payload)
            LOGGER.info('checkpoint update=%d validation=%s', update, json.dumps(row['validation']))
    atomic_json(output / 'COMPLETE.train.json', {'update': update, 'checkpoint': str(latest), 'provenance': provenance})
    return output


# Evaluation

def evaluation_protocol(config: dict[str, Any], score_sha: str, grid: Grid, metric_assets: Any) -> dict[str, Any]:
    """Everything that must be identical across arms for their metrics to be comparable."""
    settings = config['evaluation']
    return {'schema': EVALUATION_SCHEMA, 'kernel_version': KERNEL_VERSION, 'score_sha256': score_sha,
            'diffusion': config['diffusion'], 'grid': list(grid), 'nfe': len(grid) - 1,
            'guidance_scale': float(config['sampler']['guidance_scale']),
            'num_images': int(settings['num_images']), 'batch_size': int(settings['batch_size']),
            'labels': 'cyclic class-balanced 0..C-1',
            'noise': 'SeedSequence(seed, stream, batch); stream 0 latent, 1 Brownian; shared across arms',
            'pixels': 'endpoint clamped to [-1,1], stored float32 NHWC in [0,1]',
            'metric_assets': metric_assets}


def build_arm(arm: dict[str, Any], schedule: Schedule, protocol: dict[str, Any],
              device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Kernel of one arm, and the resolved description that identifies its results."""
    kind = arm['kind']
    if kind == 'fixed':
        value = float(arm['lambda'])
        return ScaffoldKernel(schedule, FixedScaffold(value)), {'kind': kind, 'lambda': value}
    if kind == 'ddpm':
        variance_type = str(arm['variance_type'])
        return DDPMKernel(schedule, variance_type=variance_type), {'kind': kind, 'variance_type': variance_type}
    if kind != 'learned':
        raise ValueError(f"arm kind must be 'fixed', 'ddpm' or 'learned', got {kind!r}")

    path = rooted_path(arm['checkpoint'])
    weights = str(arm.get('weights', 'ema'))
    scaffold, payload = load_scaffold(path, device, weights=weights)
    trained = payload['provenance']
    trained_guidance = float(payload['config']['sampler']['guidance_scale'])
    mismatches = [name for name, same in (
        ('score', trained['score_sha256'] == protocol['score_sha256']),
        ('kernel_version', trained['kernel_version'] == protocol['kernel_version']),
        ('diffusion', payload['config']['diffusion'] == protocol['diffusion']),
        ('grid', trained['grid'] == protocol['grid']),
        ('guidance_scale', trained_guidance == protocol['guidance_scale'])) if not same]
    waivable = arm.get('allow_protocol_mismatch', False) and set(mismatches) <= {'grid', 'guidance_scale'}
    if mismatches and not waivable:
        raise RuntimeError(f"arm {arm['name']!r}: checkpoint was trained under a different {mismatches}")
    state = str(arm.get('state', 'own'))
    if state == 'shuffled':
        scaffold = ShuffledStateScaffold(scaffold)
    elif state != 'own':
        raise ValueError("a learned arm's state must be 'own' or 'shuffled'")
    resolved = {'kind': kind, 'checkpoint_sha256': sha256_file(path), 'weights': weights, 'state': state,
                'update': int(payload['update']), 'training_seed': int(payload['config']['training']['seed']),
                'trained_grid': trained['grid'],
                'trained_guidance_scale': trained_guidance, 'protocol_mismatch': mismatches}
    return ScaffoldKernel(schedule, scaffold), resolved


def to_unit_nhwc(images: torch.Tensor) -> np.ndarray:
    """[-1, 1] NCHW endpoint -> clamped float32 NHWC in [0, 1], the stored sample format."""
    values = ((images.clamp(-1, 1) + 1) * 0.5).permute(0, 2, 3, 1).cpu().numpy()
    return np.ascontiguousarray(values.astype(np.float32, copy=False))


def _is_valid_shard(path: Path, shape: tuple[int, ...]) -> bool:
    try:
        return np.load(path, mmap_mode='r').shape == shape
    except (FileNotFoundError, ValueError, OSError):
        return False


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def sample_arm(directory: Path, kernel: nn.Module, epsilon_fn: EpsilonFn, labels: np.ndarray, grid: Grid, *,
               seed: int, batch_size: int, image_size: int, device: torch.device) -> Path:
    """Generate, or finish generating, one arm; returns its assembled samples file.

    Besides the samples, ``variance.json`` records for every stochastic edge the
    requested multiplier, the effective one after the budget, and the clipped
    fraction, averaged over all samples: a nominal lambda above an edge's
    budget is not the variance that was actually used.
    """
    samples = directory / 'samples.npy'
    if samples.is_file():
        return samples
    schedule = kernel.schedule
    shards, records = [], []
    for batch, start in enumerate(range(0, len(labels), batch_size)):
        y = torch.as_tensor(labels[start:start + batch_size], device=device, dtype=torch.long)
        shard = directory / 'batches' / f'batch_{batch:06d}.npy'
        record_path = directory / 'variance' / f'batch_{batch:06d}.json'
        record = _read_json(record_path)
        if record is None or not _is_valid_shard(shard, (len(y), image_size, image_size, 3)):
            rows: list[dict[str, Any]] = []

            def observe(t: int, s: int, transition: Transition) -> None:
                if s == CLEAN:
                    return
                state = transition.mean
                base = posterior_variance(schedule.alpha_bar_at(torch.full_like(y, t), state),
                                          schedule.alpha_bar_at(torch.full_like(y, s), state))
                rows.append({'edge': [t, s],
                             'lambda_requested_mean': float((transition.requested / base).mean()),
                             'lambda_effective_mean': float((transition.variance / base).mean()),
                             'variance_clip_fraction': float(transition.capped.float().mean())})

            initial = torch.randn((len(y), 3, image_size, image_size), device=device,
                                  generator=stream_generator(device, seed, LATENT_STREAM, batch))
            endpoint = sample(epsilon_fn, kernel, y, initial, grid, observer=observe,
                              generator=stream_generator(device, seed, BROWNIAN_STREAM, batch))
            record = {'samples': len(y), 'edges': rows}
            atomic_numpy(shard, to_unit_nhwc(endpoint))
            atomic_json(record_path, record)
        shards.append(shard)
        records.append(record)
    total = sum(record['samples'] for record in records)
    edges = [{'edge': first['edge'],
              **{key: sum(record['samples'] * record['edges'][index][key] for record in records) / total
                 for key in ('lambda_requested_mean', 'lambda_effective_mean', 'variance_clip_fraction')}}
             for index, first in enumerate(records[0]['edges'])]
    atomic_json(directory / 'variance.json', {'samples': total, 'edges': edges})
    atomic_numpy(samples, np.concatenate([np.load(shard) for shard in shards]))
    return samples


def summarize(rows: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """Mean and sample standard deviation across seeds of every scalar metric."""
    def scalars(value: Any, prefix: str = '') -> dict[str, float]:
        if isinstance(value, dict):
            return {key: item for name, child in value.items()
                    for key, item in scalars(child, f'{prefix}{name}.').items()}
        return {prefix[:-1]: float(value)}

    summary = {}
    for arm, by_seed in rows.items():
        flat = [scalars(metrics) for metrics in by_seed.values()]
        if flat:
            summary[arm] = {key: {'mean': float(np.mean([row[key] for row in flat])),
                                  'std': float(np.std([row[key] for row in flat], ddof=1)) if len(flat) > 1 else None,
                                  'seeds': len(flat)}
                            for key in flat[0]}
    return summary


def paired_differences(rows: dict[str, dict[int, dict[str, Any]]], reference: str) -> dict[str, Any]:
    """arm - reference for every scalar metric, paired by evaluation seed.

    Arms of one seed share latents and Brownian draws, so the per-seed
    difference removes the seed's sampling noise. The unit of replication is
    the seed: FID is a statistic of a whole sample set and has no per-image
    difference. ``std`` is the sample standard deviation of the per-seed
    differences; with few seeds it is a rough indication, not a test.
    """
    def scalars(value: Any, prefix: str = '') -> dict[str, float]:
        if isinstance(value, dict):
            return {key: item for name, child in value.items()
                    for key, item in scalars(child, f'{prefix}{name}.').items()}
        return {prefix[:-1]: float(value)}

    baseline = {seed: scalars(metrics) for seed, metrics in rows[reference].items()}
    result = {}
    for arm, by_seed in rows.items():
        seeds = sorted(set(by_seed) & set(baseline))
        if arm == reference or not seeds:
            continue
        flat = {seed: scalars(by_seed[seed]) for seed in seeds}
        result[arm] = {}
        for key in flat[seeds[0]]:
            if not all(key in baseline[seed] for seed in seeds):
                continue
            deltas = [flat[seed][key] - baseline[seed][key] for seed in seeds]
            result[arm][key] = {'mean': float(np.mean(deltas)),
                                'std': float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
                                'seeds': len(deltas), 'per_seed': dict(zip(map(str, seeds), deltas))}
    return result


def collect_summary(output: Path, protocol: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Summary of every scored (seed, arm) cell on disk that belongs to ``protocol``.

    Built from the stored reports, not from the arms of the current call, so a
    partial run (``--arms``) extends the summary instead of replacing it.
    """
    digest = canonical_hash(protocol)
    rows: dict[str, dict[int, dict[str, Any]]] = {}
    arms: dict[str, dict[str, Any]] = {}
    variance: dict[str, dict[str, float]] = {}
    for report_path in sorted(output.glob('seed_*/*/metrics.json')):
        report = _read_json(report_path)
        if report is None or report['binding']['protocol_sha256'] != digest:
            continue
        name = report_path.parent.name
        rows.setdefault(name, {})[int(report['binding']['seed'])] = report['metrics']
        arms[name] = report['binding']['arm']
        record = _read_json(report_path.parent / 'variance.json')
        if record is not None and record['edges']:
            variance[name] = {key: float(np.mean([edge[key] for edge in record['edges']]))
                              for key in ('lambda_requested_mean', 'lambda_effective_mean', 'variance_clip_fraction')}
    summary: dict[str, Any] = {'protocol_sha256': digest, 'version': __version__, 'arms': arms,
                               'variance': variance, 'per_seed': rows, 'across_seeds': summarize(rows)}
    paired = {}
    reference = settings.get('reference_arm', 'ddim')
    if reference in rows:
        paired[reference] = paired_differences(rows, reference)
    # The best fixed-lambda arm is chosen on these same seeds, so differences against it are optimistic for it.
    metric = settings.get('selection_metric', 'FID.Overall')
    fixed = {name: summary['across_seeds'][name][metric]['mean'] for name in rows
             if arms[name]['kind'] == 'fixed' and metric in summary['across_seeds'].get(name, {})}
    if fixed:
        best = min(fixed, key=fixed.get)
        summary['best_fixed_arm'] = {'name': best, 'selection_metric': metric, 'selected_on_reported_seeds': True}
        if best not in paired:
            paired[best] = paired_differences(rows, best)
    summary['paired_differences'] = paired
    return summary


def evaluate(config: dict[str, Any], device: torch.device, *, arms: list[str] | None = None,
             sample_only: bool = False, output: Path | None = None,
             epsilon: tuple[EpsilonFn, str] | None = None, metric_fn: MetricFn | None = None) -> Path:
    """Sample every (seed, arm) cell and score it; returns the evaluation directory.

    ``arms`` restricts the run to the named arms. ``output``, ``epsilon`` and
    ``metric_fn`` replace the configured locations and assets (CPU tests).
    """
    settings = config['evaluation']
    seeds = [int(seed) for seed in settings['seeds']]
    defined = [arm['name'] for arm in settings['arms']]
    if arms is not None and set(arms) - set(defined):
        raise ValueError(f'unknown arms: {sorted(set(arms) - set(defined))}')
    selected = [arm for arm in settings['arms'] if arms is None or arm['name'] in arms]
    names = [arm['name'] for arm in selected]
    if int(settings['num_images']) % int(config['dataset']['num_classes']):
        raise ValueError('evaluation.num_images must be a multiple of the class count')
    if not seeds or len(set(seeds)) != len(seeds) or not names or len(set(defined)) != len(defined):
        raise ValueError('evaluation needs distinct seeds and at least one arm, with unique arm names')

    output = output_directory(str(settings['name'])) if output is None else Path(output)
    schedule = Schedule.from_config(config)
    grid = build_grid(len(schedule), config['sampler']['grid'])
    epsilon_fn, score_sha = frozen_epsilon(config, device) if epsilon is None else epsilon
    if metric_fn is None:
        metrics_root = rooted_path(settings['metrics_root'])
        assets: Any = metric_asset_binding(metrics_root)
        if settings.get('classifier'):
            assets['classifier_sha256'] = sha256_file(Path(settings['classifier']['torchscript']))

        def metric_fn(samples: Path, labels: np.ndarray) -> dict[str, Any]:
            return compute_metrics(samples, labels, settings, metrics_root, device)
    else:
        assets = 'supplied metric function'
    protocol = evaluation_protocol(config, score_sha, grid, assets)
    protocol_path = output / 'PROTOCOL.json'
    if protocol_path.is_file() and json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
        raise RuntimeError(f'{output} holds results for a different protocol; change evaluation.name')
    atomic_json(protocol_path, protocol)

    labels = np.arange(int(settings['num_images']), dtype=np.int64) % int(config['dataset']['num_classes'])
    kernels = [(arm['name'], *build_arm(arm, schedule, protocol, device)) for arm in selected]
    for seed in seeds:
        for name, kernel, resolved in kernels:
            directory = output / f'seed_{seed}' / name
            binding = {'protocol_sha256': canonical_hash(protocol), 'seed': seed, 'arm': resolved}
            binding_path = directory / 'ARM.json'
            if binding_path.is_file() and json.loads(binding_path.read_text(encoding='utf-8')) != binding:
                raise RuntimeError(f'{directory} was produced by a different arm definition; rename the arm')
            atomic_json(binding_path, binding)
            samples = sample_arm(directory, kernel, epsilon_fn, labels, grid, seed=seed,
                                 batch_size=int(settings['batch_size']),
                                 image_size=int(config['dataset']['image_size']), device=device)
            LOGGER.info('sampled seed=%d arm=%s', seed, name)
            if sample_only:
                continue
            report_path = directory / 'metrics.json'
            digest = sha256_file(samples)
            report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.is_file() else None
            if report is None or report['samples_sha256'] != digest:
                report = {'samples_sha256': digest, 'binding': binding, 'metrics': metric_fn(samples, labels)}
                atomic_json(report_path, report)
            LOGGER.info('seed=%d arm=%s metrics=%s', seed, name, json.dumps(report['metrics']))
    if not sample_only:
        atomic_json(output / 'summary.json', collect_summary(output, protocol, settings))
    return output


# Diagnostics

def diagnose(config: dict[str, Any], device: torch.device, *, checkpoint: Path | None = None,
             weights: str = 'ema', samples: int = 256, output: Path | None = None,
             epsilon: tuple[EpsilonFn, str] | None = None, data: Dataset | None = None) -> Path:
    """Edge-level diagnostics of the configured objective; returns the JSON report path.

    Scores every edge of the sampling grid on one fixed class-balanced batch of
    training images, with the scaffold of ``checkpoint`` or, without one, the
    freshly initialised scaffold that training would start from.
    """
    epsilon_fn, score_sha = frozen_epsilon(config, device) if epsilon is None else epsilon
    schedule = Schedule.from_config(config)
    grid = build_grid(len(schedule), config['sampler']['grid'])
    loss = build_loss(config['loss'], schedule, grid)
    if not hasattr(loss, 'edge_diagnostics'):
        raise ValueError(f"objective {config['loss']['kind']!r} does not provide edge_diagnostics")
    images, indices, _ = load_training_data(config) if data is None else data
    rng = np.random.default_rng(int(config['validation']['seed']))
    classes = np.arange(samples) % len(indices)
    chosen = np.array([rng.choice(indices[class_id]) for class_id in classes])
    if checkpoint is None:
        torch.manual_seed(int(config['training']['seed']))
        scaffold, tag = build_scaffold(config['scaffold'], len(indices)).to(device), 'initial'
    else:
        scaffold, payload = load_scaffold(Path(checkpoint), device, weights=weights)
        scaffold.requires_grad_(True)
        tag = f"update_{payload['update']}_{weights}"
    report = loss.edge_diagnostics(
        epsilon_fn, ScaffoldKernel(schedule, scaffold), to_pixels(images[chosen], device),
        torch.as_tensor(classes, device=device, dtype=torch.long),
        generator=torch.Generator(device=device).manual_seed(int(config['validation']['seed'])))
    report.update(version=__version__, objective=config['loss']['kind'], scaffold=tag, samples=samples,
                  grid=list(grid), score_sha256=score_sha, config_sha256=canonical_hash(training_identity(config)))
    output = output_directory(f"{config['training']['name']}_diagnostics") if output is None else Path(output)
    atomic_json(output / f'{tag}.json', report)
    summary = report['summary']
    LOGGER.info('dominant edge %s carries %.1f%% of the full bound; zero-gradient edges: %s; edges preferring lambda<1: %s',
                summary['dominant_edge'], 100 * summary['dominant_edge_share_of_full_bound'],
                summary['zero_gradient_edges'], summary['edges_preferring_lambda_below_one'])
    return output / f'{tag}.json'


# Preflight (Gate D)

def _vendor_diffusion() -> Any:
    """The pinned CCUA-DDPM samplers, used only as a parity reference."""
    root = str(ROOT / 'third_party' / 'CCUA-DDPM')
    sys.path.insert(0, root)
    try:
        return importlib.import_module('diffusion')
    finally:
        sys.path.remove(root)
        sys.modules.pop('diffusion', None)


def noise_prediction_verdict(index: int, steps: int, mse: float, wrong_timestep_mse: dict[str, float],
                             trivial_mse: float) -> dict[str, Any]:
    """Verdict of ``score_noise_prediction`` for one noise level; the criteria are explained there."""
    data_regime = index <= 6 * steps // 10
    return {'mse': mse, 'wrong_timestep_mse': wrong_timestep_mse, 'trivial_mse': trivial_mse,
            'regime': 'data' if data_regime else 'near_noise',
            'beats_wrong_timesteps': all(mse < value for value in wrong_timestep_mse.values()),
            'quality_ok': mse < trivial_mse if data_regime else mse < NEAR_NOISE_FACTOR * trivial_mse}


def preflight(config: dict[str, Any], device: torch.device, *, num_images: int = 1000, output: Path | None = None,
              score: tuple[nn.Module, str] | None = None, data: Dataset | None = None,
              metric_fn: MetricFn | None = None) -> tuple[Path, bool]:
    """Check that this code reproduces the baselines with the real pretrained score.

    Writes ``preflight.json`` with one pass/fail entry per check and returns
    (report path, overall result). Nothing is trained. A failed check is
    reported and the remaining checks still run; checks that depend on a failed
    one fail with the missing prerequisite as their reason.

    ``score``, ``data`` and ``metric_fn`` replace the real assets in CPU tests;
    the Inception check runs only with the real metric assets.
    """
    output = output_directory('preflight') if output is None else Path(output)
    diffusion, guidance = config['diffusion'], float(config['sampler']['guidance_scale'])
    schedule = Schedule.from_config(config)
    state: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    tolerance = 5e-3

    def check(name: str, function: Callable[[], dict[str, Any]]) -> None:
        try:
            detail = function()
            passed = bool(detail.pop('passed'))
        except Exception as error:  # a failed check must not hide the others
            passed, detail = False, {'error': f'{type(error).__name__}: {error}'}
        checks.append({'name': name, 'passed': passed, **detail})
        LOGGER.info('preflight %-24s %s %s', name, 'PASS' if passed else 'FAIL', json.dumps(detail, default=str))

    def need(key: str, prerequisite: str) -> Any:
        if key not in state:
            raise RuntimeError(f'prerequisite check {prerequisite!r} failed')
        return state[key]

    def epsilon_at(scale: float) -> FrozenEpsilon:
        return FrozenEpsilon(need('model', 'score_checkpoint'), guidance_scale=scale,
                             microbatch=int(config['model']['microbatch']))

    def score_checkpoint() -> dict[str, Any]:
        state['model'], state['sha'] = load_frozen_score(config, device) if score is None else score
        return {'passed': True, 'sha256': state['sha'],
                'parameters': sum(parameter.numel() for parameter in state['model'].parameters())}

    def noise_schedule() -> dict[str, Any]:
        reference = _vendor_diffusion().GaussianDiffusionSampler(
            nn.Identity(), diffusion['beta_1'], diffusion['beta_T'], diffusion['steps'], num_class=1)
        return {'passed': torch.equal(reference.alphas_bar, schedule.alpha_bar), 'steps': len(schedule)}

    def dataset_split() -> dict[str, Any]:
        images, indices, counts = load_training_data(config) if data is None else data
        pixels = to_pixels(images[:256], device)
        state.update(images=images, indices=indices)
        detail = {'images': len(images), 'largest_class': int(counts.max()), 'smallest_class': int(counts.min()),
                  'images_sha256': array_sha256(images), 'pixel_range': [float(pixels.min()), float(pixels.max())]}
        passed = images.dtype == np.uint8 and float(pixels.min()) >= -1 and float(pixels.max()) <= 1
        if data is None:
            classes, factor = int(config['dataset']['num_classes']), float(config['dataset']['imbalance_factor'])
            expected = [int(500 * factor ** (index / (classes - 1.0))) for index in range(classes)]
            passed = passed and counts.astype(int).tolist() == expected
        return {'passed': passed, **detail}

    def panel(count: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        indices = need('indices', 'dataset_split')
        classes = np.arange(count) * 13 % len(indices)
        chosen = [indices[class_id][0] for class_id in classes]
        return to_pixels(state['images'][chosen], device), torch.as_tensor(classes, device=device, dtype=torch.long)

    def score_noise_prediction() -> dict[str, Any]:
        """Timestep convention end to end: index i is the noise level alpha_bar[i].

        Two questions per noise level, each against a reference that is meaningful there:
          * convention: theta told the true index must beat theta told an index 15 % of the schedule
            away, in every direction that stays inside the schedule;
          * quality: where the data matter (index <= 0.6 T) theta must beat the trivial predictor
            eps = sqrt(1 - ab) x_t, which is exact only for N(0, I) data. Close to pure noise that
            predictor is itself near-optimal (its error is O(ab)), so a network cannot be asked to beat
            it; there theta must stay within NEAR_NOISE_FACTOR of it. A wrong timestep convention
            misses this by orders of magnitude.
        This check sees offsets of roughly 10 % of the schedule or more. Off-by-one conventions are
        caught by the parity checks against the vendor samplers, not here.
        """
        x0, labels = panel(32)
        epsilon, rows = epsilon_at(0.0), {}
        steps = len(schedule)
        shift = 3 * steps // 20
        for index in (steps // 20, 3 * steps // 10, 6 * steps // 10, 19 * steps // 20):
            alpha = schedule.alpha_bar_at(torch.full_like(labels, index), x0)
            noise = torch.randn(x0.shape, device=device, generator=torch.Generator(device).manual_seed(index))
            x_t = forward_sample(x0, alpha, noise)

            def mse(told: int) -> float:
                return float((epsilon(x_t, torch.full_like(labels, told), labels) - noise).square().mean())

            error = mse(index)
            wrong = {str(told): mse(told) for told in (index - shift, index + shift) if 0 <= told < steps}
            trivial = float((x_t * (1 - alpha).sqrt() - noise).square().mean())
            rows[str(index)] = noise_prediction_verdict(index, steps, error, wrong, trivial)
        return {'passed': all(row['beats_wrong_timesteps'] and row['quality_ok'] for row in rows.values()),
                'near_noise_factor': NEAR_NOISE_FACTOR, 'timesteps': rows}

    def ddim_parity() -> dict[str, Any]:
        """lambda = 0 on the vendor stride-10 grid against the vendor DDIM sampler.

        Primary criterion, per step: fed the vendor's own states, each of our edges must reproduce the
        vendor's next state to STEP_TOLERANCE. This isolates the update formula from the dynamics.

        Secondary criterion, endpoint: two float32 implementations of the same formula differ by
        round-off at every step, and a guided chain amplifies such perturbations. The amplification
        is measured here, not assumed: our own endpoint is recomputed from an initial state nudged
        by 1e-6. The vendor-vs-ours endpoint gap must not exceed ENDPOINT_AMPLIFICATION times that
        sensitivity (one round-off-sized kick per step over ~100 steps), with an absolute floor.
        """
        _, labels = panel()
        initial = torch.randn((len(labels), 3, int(config['dataset']['image_size']), int(config['dataset']['image_size'])),
                              device=device, generator=torch.Generator(device).manual_seed(0))
        nudge = 1e-6 * torch.randn(initial.shape, device=device, generator=torch.Generator(device).manual_seed(5))
        grid = stride_grid(len(schedule), 10)
        kernel = ScaffoldKernel(schedule, FixedScaffold(0.0))
        rows = {}
        for scale in sorted({0.0, guidance, 1.5}):
            reference = _vendor_diffusion().GaussianDiffusionSamplerDDIM(
                need('model', 'score_checkpoint'), diffusion['beta_1'], diffusion['beta_T'], diffusion['steps'],
                omega=scale, cond=True).to(device)
            epsilon = epsilon_at(scale)
            step_error, state = 0.0, initial.clone()
            with torch.no_grad():
                for current, following in zip(grid, grid[1:]):
                    t, s = torch.full_like(labels, current), torch.full_like(labels, following)
                    reference.omega_t = scale
                    expected_next = reference.p_mean_variance(x_t=state, t=t, y=labels, method='ddim', skip=10)[0]
                    step_error = max(step_error, float((kernel(state, epsilon(state, t, labels), t, s, labels).mean
                                                        - expected_next).abs().max()))
                    state = expected_next
                expected = reference(initial, labels, method='ddim', skip=10)

            def run(start: torch.Tensor) -> torch.Tensor:
                return sample(epsilon, kernel, labels, start, grid, generator=torch.Generator(device).manual_seed(1)).clamp(-1, 1)

            ours = run(initial)
            endpoint = float((ours - expected).abs().max())
            sensitivity = float((ours - run(initial + nudge)).abs().max())
            allowed = max(ENDPOINT_FLOOR, ENDPOINT_AMPLIFICATION * sensitivity)
            rows[str(scale)] = {'max_step_error': step_error, 'endpoint_error': endpoint,
                                'endpoint_sensitivity_to_1e-6_nudge': sensitivity, 'endpoint_allowed': allowed,
                                'step_ok': step_error < STEP_TOLERANCE, 'endpoint_ok': endpoint <= allowed}
        return {'passed': all(row['step_ok'] and row['endpoint_ok'] for row in rows.values()),
                'step_tolerance': STEP_TOLERANCE, 'by_guidance': rows}

    def ddpm_parity() -> dict[str, Any]:
        """Last 200 ancestral steps against the vendor mean / variance, same start and noise."""
        x0, labels = panel()
        start_index = min(199, len(schedule) - 1)
        start = forward_sample(x0, schedule.alpha_bar_at(torch.full_like(labels, start_index), x0),
                               torch.randn(x0.shape, device=device, generator=torch.Generator(device).manual_seed(2)))
        errors = {}
        for variance_type in ('fixedlarge', 'fixedsmall'):
            reference = _vendor_diffusion().GaussianDiffusionSampler(
                need('model', 'score_checkpoint'), diffusion['beta_1'], diffusion['beta_T'], diffusion['steps'],
                num_class=int(config['dataset']['num_classes']), var_type=variance_type).to(device)
            generator = torch.Generator(device).manual_seed(3)
            expected = start.clone()
            with torch.no_grad():
                for step in range(start_index, -1, -1):
                    mean, log_variance = reference.p_mean_variance(expected, torch.full_like(labels, step),
                                                                   y=labels, omega=guidance)
                    noise = torch.randn(expected.shape, device=device, generator=generator) if step > 0 else 0
                    expected = mean + torch.exp(0.5 * log_variance) * noise
            ours = sample(epsilon_at(guidance), DDPMKernel(schedule, variance_type=variance_type), labels, start,
                          tuple(range(start_index, CLEAN - 1, -1)), generator=torch.Generator(device).manual_seed(3))
            errors[variance_type] = float((ours - expected).abs().max())
        return {'passed': max(errors.values()) < tolerance, 'max_abs_error_by_variance': errors, 'tolerance': tolerance}

    def metric_assets() -> dict[str, Any]:
        return {'passed': True, **metric_asset_binding(rooted_path(config['evaluation']['metrics_root']))}

    def inception_preprocessing() -> dict[str, Any]:
        """Our Inception pipeline on real CIFAR-100 images must reproduce the stored reference features."""
        import torchvision.datasets as datasets
        root = rooted_path(config['evaluation']['metrics_root'])
        manifest = json.loads((root / MANIFEST).read_text(encoding='utf-8'))
        dataset = datasets.CIFAR100(root=str(rooted_path(config['dataset']['root'])), train=True, download=False)
        if not np.array_equal(np.load(root / manifest['labels_file']), np.asarray(dataset.targets)):
            raise RuntimeError('reference features are not in CIFAR-100 training order; cannot compare row by row')
        images = np.asarray(dataset.data[:256], dtype=np.float32) / 255.0
        features, _ = inception_statistics(images, batch_size=int(config['evaluation']['inception_batch_size']), device=device)
        reference = np.load(root / manifest['feature_file'], mmap_mode='r')[:256]
        error = float(np.abs(features - reference).max() / np.abs(reference).max())
        return {'passed': error < 1e-2, 'max_relative_feature_error': error, 'images': len(images)}

    def baseline_panel() -> dict[str, Any]:
        settings = copy.deepcopy(config)
        names = [arm['name'] for arm in settings['evaluation']['arms']
                 if arm['kind'] == 'ddpm' or (arm['kind'] == 'fixed' and float(arm['lambda']) in (0.0, 1.0))]
        settings['evaluation'].update(num_images=num_images, seeds=[int(settings['evaluation']['seeds'][0])])
        directory = evaluate(settings, device, arms=names, output=output / 'panel',
                             epsilon=(epsilon_at(guidance), need('sha', 'score_checkpoint')), metric_fn=metric_fn)
        summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        values = [entry['mean'] for arm in summary['across_seeds'].values() for entry in arm.values()]
        return {'passed': bool(values) and bool(np.isfinite(values).all()), 'arms': names, 'num_images': num_images,
                'metrics': summary['per_seed'], 'variance': summary['variance']}

    check('score_checkpoint', score_checkpoint)
    check('noise_schedule', noise_schedule)
    check('dataset_split', dataset_split)
    check('score_noise_prediction', score_noise_prediction)
    check('ddim_parity', ddim_parity)
    check('ddpm_parity', ddpm_parity)
    if metric_fn is None:
        check('metric_assets', metric_assets)
        check('inception_preprocessing', inception_preprocessing)
    check('baseline_panel', baseline_panel)
    passed = all(entry['passed'] for entry in checks)
    report = {'passed': passed, 'version': __version__, 'kernel_version': KERNEL_VERSION,
              'config_sha256': canonical_hash(config), 'source_sha256': source_hashes(), 'checks': checks}
    atomic_json(output / 'preflight.json', report)
    return output / 'preflight.json', passed


def main() -> None:
    parser = argparse.ArgumentParser(description='Learned DDIM-family scaffold on a frozen DDPM.')
    parser.add_argument('command', choices=('preflight', 'diagnose', 'train', 'evaluate'))
    parser.add_argument('--config', type=Path, default=SCAFFOLD_ROOT / 'config.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                        help='override a config entry, e.g. --set sampler.grid.nfe=50')
    parser.add_argument('--resume', action='store_true', help='train: continue from latest.pt')
    parser.add_argument('--arms', nargs='+', help='evaluate: run only these arms')
    parser.add_argument('--sample-only', action='store_true', help='evaluate: generate samples, skip metrics')
    parser.add_argument('--checkpoint', type=Path, help='diagnose: trained scaffold (default: initial scaffold)')
    parser.add_argument('--weights', choices=('ema', 'online'), default='ema', help='diagnose: checkpoint weights')
    parser.add_argument('--samples', type=int, default=256, help='diagnose: training images to score')
    parser.add_argument('--num-images', type=int, default=1000, help='preflight: samples per baseline arm')
    args = parser.parse_args()
    if args.command != 'evaluate' and (args.arms or args.sample_only):
        parser.error('--arms and --sample-only belong to evaluate')
    if args.command != 'train' and args.resume:
        parser.error('--resume belongs to train; evaluate always resumes')
    if args.command != 'diagnose' and args.checkpoint:
        parser.error('--checkpoint belongs to diagnose')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    config = load_config(args.config, args.overrides)
    device = torch.device(args.device)
    if args.command == 'preflight':
        output, passed = preflight(config, device, num_images=args.num_images)
        LOGGER.info('preflight %s: %s', 'PASSED' if passed else 'FAILED - do not train', output)
        sys.exit(0 if passed else 1)
    if args.command == 'diagnose':
        output = diagnose(config, device, checkpoint=args.checkpoint, weights=args.weights, samples=args.samples)
    elif args.command == 'train':
        output = train(config, device, resume=args.resume)
    else:
        output = evaluate(config, device, arms=args.arms, sample_only=args.sample_only)
    LOGGER.info('%s complete: %s', args.command, output)


if __name__ == '__main__':
    main()
