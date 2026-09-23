# Long-tail diffusion baseline runpack

This repository runs every supported comparison through one pinned
`third_party/CCUA-DDPM/` runtime. DDPM, CBDM, T2H, and CCUA therefore share the
same dataset split, U-Net, optimizer, checkpoint format, sampler, and evaluator.

Research work that is not part of this baseline contract is preserved on the
local branch `archive/research-wip-20260922`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `configs/` | Baseline campaign and server definitions |
| `ltx/` | Planning, preflight, scheduling, resume, and command generation |
| `third_party/CCUA-DDPM/` | The only training and sampling backbone |
| `tools/evaluate_ccua.py` | Shared CIFAR evaluator |
| `scripts/` | Bootstrap, launch, status, resume, stop, and Slurm entrypoints |
| `tests/` | Baseline unit and regression tests |

`data/`, `runs/`, `outputs/`, W&B files, metric arrays, checkpoints, and logs
are local artifacts. They are intentionally ignored by Git.

## Supported objectives

| Objective | Runtime behavior |
| --- | --- |
| DDPM | Native epsilon-prediction loss |
| CBDM | Native class-balancing loss with `cb_tau` |
| T2H | Native transfer target on the shared CCUA host |
| CCUA | Native alignment and unconditional contrastive losses |

Any other objective is rejected by both the adapter and preflight checks.

## Locked CIFAR-100-LT protocol

- CIFAR-100-LT IF100, 32×32, split seed 0;
- 300,000 updates, batch 64, learning rate `2e-4`, T=1000, EMA `0.9999`;
- 50,000 class-uniform generated labels;
- DDIM-100 with guidance scale 1.5;
- shared FID, KID, IS, precision/recall, and Many/Medium/Few evaluation.

The main campaign contains DDPM/CBDM/CCUA × seeds 0, 1, and 2. The seed-0
comparison config also contains T2H.

## Environment

Create `.env.local` from `.env.example` and keep it local. It is never tracked.
On the server:

```bash
cd /home/nvidia-lab/ai4life/annd/respi/longtail_server_runpack
source .venv/bin/activate
export LTX_REPOS_ROOT="$PWD/third_party"
export LTX_DATA_ROOT="$PWD/data"
export LTX_RUNS_ROOT="/home/nvidia-lab/data_mount/longtail_server_runpack/runs"
export LTX_METRICS_ROOT="$PWD/third_party/CCUA-DDPM/stats"
```

Bootstrap and validate without launching training:

```bash
bash scripts/bootstrap.sh
python -m ltx.cli plan --config configs/native_cifar100_if100.yaml
python -m ltx.cli preflight --config configs/native_cifar100_if100.yaml
```

Launch the main campaign:

```bash
bash scripts/run_native_cifar_c100.sh
```

Launch the smaller baseline/T2H seed-0 comparison:

```bash
bash scripts/run_ccua_ddpm_cbdm.sh \
  configs/native_cifar100_if100_ddpm_cbdm_seed0.yaml
```

## Recovery and evaluation

Checkpoints are atomically replaced and their interval is capped by the server
retry policy. If a worker is killed before writing terminal state, the
scheduler records the exit and retries from the newest complete checkpoint.

```bash
python -m ltx.cli status --config configs/native_cifar100_if100.yaml
tail -f runs/<campaign>/worker_*.launcher.log
```

The evaluator may write a fast preview before the canonical detailed result.
Generated Inception features are cached beside the run and reused only when the
sample artifact identity still matches.

## Development checks

```bash
python -m pytest -q
git diff --check
```

Keep baseline behavior in the shared runtime, expose configuration through the
CCUA adapter, and add a focused regression test for every operational change.
