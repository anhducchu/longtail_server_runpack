"""Final baseline study at CFG 1.5: development sweep, selection, frozen 5k evaluation panel.

    python tools/baseline_cfg15.py [--output-root scaffold/outputs] [--device cuda]

1. Development sweep (seeds 100-102): lambda in {0, 0.01, 0.04, 0.09, 0.25, 1} on the vendor uniform
   grid (stride 50 / 20: 19 / 49 score evaluations, state returned at index 0) and on the quadratic
   grid (20 / 50 score evaluations, ending with the clean edge).
2. Selection per budget (about 20 and about 50 NFE) by mean Overall FID on the development seeds:
   the best deterministic configuration (lambda = 0) and the best stochastic one (lambda > 0).
3. Frozen panel: the selected configurations on evaluation seeds 200-202, never used for selection.

Every panel is an ordinary `scaffold.main evaluate` run, so it is resumable and paired by seed.
Nothing is trained. `baseline_cfg15.json` holds the selection and the frozen table.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scaffold import __version__  # noqa: E402
from scaffold.main import evaluate  # noqa: E402
from scaffold.sampling import build_grid  # noqa: E402
from scaffold.utils import SCAFFOLD_ROOT, atomic_json, load_config, source_hashes  # noqa: E402

GUIDANCE = 1.5
LAMBDAS = (0.0, 0.01, 0.04, 0.09, 0.25, 1.0)
DEVELOPMENT_SEEDS, EVALUATION_SEEDS = [100, 101, 102], [200, 201, 202]
GRIDS = {20: {'uniform_s50': {'stride': 50}, 'quadratic20': {'nfe': 20, 'spacing': 'quadratic'}},
         50: {'uniform_s20': {'stride': 20}, 'quadratic50': {'nfe': 50, 'spacing': 'quadratic'}}}


def arm_name(lam: float) -> str:
    return 'ddim' if lam == 0 else f'lam_{lam:g}'


def panel(base: dict, grid: dict, name: str, seeds: list[int], lambdas, output_root: Path, device) -> dict:
    config = copy.deepcopy(base)
    config['sampler'] = {'grid': grid, 'guidance_scale': GUIDANCE}
    config['model']['microbatch'] = 200  # throughput only
    config['evaluation'].update(name=name, seeds=seeds, reference_arm='ddim',
                                arms=[{'name': arm_name(lam), 'kind': 'fixed', 'lambda': lam} for lam in lambdas])
    directory = evaluate(config, device, output=output_root / name)
    return json.loads((directory / 'summary.json').read_text())


def row(summary: dict, arm: str, grid_name: str, grid: dict, steps: int) -> dict:
    stats = summary['across_seeds'][arm]
    evaluations = len(build_grid(steps, grid)) - 1
    return {'grid': grid_name, 'grid_spec': grid, 'arm': arm, 'lambda': summary['arms'][arm]['lambda'],
            'score_evaluations_per_image': evaluations, 'model_forwards_per_image': 2 * evaluations,
            'seeds': stats['FID.Overall']['seeds'],
            **{key: {'mean': stats[key]['mean'], 'std': stats[key]['std']}
               for key in ('FID.Overall', 'FID.Many', 'FID.Medium', 'FID.Few', 'IS.Overall.mean', 'Precision', 'Recall')
               if key in stats}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output-root', type=Path, default=SCAFFOLD_ROOT / 'outputs')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)
    base = load_config(SCAFFOLD_ROOT / 'config.json')
    steps = int(base['diffusion']['steps'])
    report = {'version': __version__, 'source_sha256': source_hashes(), 'guidance_scale': GUIDANCE,
              'num_images': base['evaluation']['num_images'], 'development_seeds': DEVELOPMENT_SEEDS,
              'evaluation_seeds': EVALUATION_SEEDS, 'selection_metric': 'FID.Overall mean over development seeds',
              'budgets': {}}
    for budget, grids in GRIDS.items():
        development = {name: panel(base, grid, f'cfg15_dev_{name}', DEVELOPMENT_SEEDS, LAMBDAS, args.output_root, device)
                       for name, grid in grids.items()}
        candidates = [row(development[name], arm, name, grids[name], steps)
                      for name in grids for arm in development[name]['across_seeds']]
        best = {'deterministic': min((c for c in candidates if c['lambda'] == 0), key=lambda c: c['FID.Overall']['mean']),
                'stochastic': min((c for c in candidates if c['lambda'] > 0), key=lambda c: c['FID.Overall']['mean'])}
        frozen = {}
        for name in sorted({choice['grid'] for choice in best.values()}):
            lambdas = sorted({choice['lambda'] for choice in best.values() if choice['grid'] == name} | {0.0})
            summary = panel(base, grids[name], f'baseline5k_cfg15_{name}', EVALUATION_SEEDS, lambdas, args.output_root, device)
            for role, choice in best.items():
                if choice['grid'] == name:
                    frozen[role] = row(summary, choice['arm'], name, grids[name], steps)
                    paired = summary['paired_differences'].get('ddim', {}).get(choice['arm'], {})
                    frozen[role]['paired_delta_vs_same_grid_ddim'] = {key: paired[key] for key in ('FID.Overall', 'FID.Few') if key in paired}
        report['budgets'][str(budget)] = {'development_candidates': candidates, 'selected_on_development': best,
                                          'frozen_evaluation': frozen}
        atomic_json(args.output_root / 'baseline_cfg15.json', report)
        print(f'budget {budget} done', flush=True)


if __name__ == '__main__':
    main()
