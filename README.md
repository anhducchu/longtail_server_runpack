# Long-tail diffusion baseline runpack

This repository runs DDPM, CBDM, T2H, and CCUA through one pinned
`third_party/CCUA-DDPM/` host. The four objectives share the CIFAR-100-LT
split, U-Net, optimizer, checkpoint format, sampler, and evaluator. Any other
objective or runtime host is rejected before launch.

Archived research code is preserved on the local branch
`archive/research-wip-20260922`; it is not part of the `main` runtime.

## Start here

- To run an experiment, start at `scripts/run.sh`.
- To read the runner, start at `src/ltx/__main__.py`, then `src/ltx/cli.py`.
- To inspect training or sampling, read `third_party/CCUA-DDPM/main.py` only
  after the runner flow is clear.

```text
scripts/run.sh
  -> python -m ltx
  -> cli.py
  -> config.py
  -> scheduler.py
  -> worker.py
  -> adapters/ccua.py
  -> third_party/CCUA-DDPM/main.py
  -> tools/evaluate_ccua.py
```

`src/ltx` is deliberately a small flat package: each file owns one part of
the execution flow, so there are no extra framework or service layers to
traverse.

## Repository map

| Path | Purpose |
| --- | --- |
| `configs/` | Main 9-task campaign, seed-0 T2H comparison, smoke, and server policy |
| `src/ltx/` | CLI, config loading, preflight, scheduling, workers, retries, and metrics |
| `third_party/CCUA-DDPM/` | Baseline-only CIFAR-100 trainer, sampler, and metric host |
| `tools/` | Metric-asset preparation, shared evaluation, and final report |
| `scripts/` | One local runner, bootstrap, and Slurm entrypoint |
| `tests/` | Baseline unit and regression suite |

Data, checkpoints, generated arrays, logs, reports, W&B state, and metric
arrays remain local and are ignored by Git. Only metric manifests are tracked.

## Locked protocol

- CIFAR-100-LT IF100, 32×32, split seed 0;
- 300,000 updates, batch 64, learning rate `2e-4`, T=1000, EMA `0.9999`;
- 50,000 class-uniform samples;
- DDIM-100, guidance scale 1.5;
- shared FID, KID, IS, F8/F1/8, improved precision/recall, and
  Many/Medium/Few FID.

The main config is DDPM/CBDM/CCUA × seeds 0, 1, and 2 (9 tasks). The separate
seed-0 config runs DDPM/CBDM/T2H on the same host.

## Configure and run

`.env.example` is the only tracked environment contract:

```bash
cp .env.example .env.local
# Fill WANDB_API_KEY, storage roots, and optional GPU-packing overrides.
```

Plan and preflight without launching:

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
python -m ltx plan --config configs/native_cifar100_if100.yaml
python -m ltx preflight --config configs/native_cifar100_if100.yaml
```

Run the main campaign or the seed-0 T2H comparison:

```bash
bash scripts/run.sh
bash scripts/run.sh configs/native_cifar100_if100_seed0.yaml
```

`scripts/run.sh` selects or creates the CUDA environment, bootstraps the local
venv, prepares missing CIFAR metric assets, launches the scheduler, and writes
the report. CLI flags after the optional config path pass through to `ltx run`:

```bash
bash scripts/run.sh --gpus 0,1,2,3 --per-gpu auto --jobs auto
```

For Slurm:

```bash
sbatch scripts/submit_slurm.sh
sbatch scripts/submit_slurm.sh configs/native_cifar100_if100_seed0.yaml
```

## Operate and recover

The CLI replaces per-command shell wrappers:

```bash
python -m ltx status --config configs/native_cifar100_if100.yaml
python -m ltx status --watch 20 --config configs/native_cifar100_if100.yaml
python -m ltx retry-failed --config configs/native_cifar100_if100.yaml
python -m ltx stop --config configs/native_cifar100_if100.yaml
```

Rerunning the same campaign resumes from the newest complete local checkpoint.
To resume one row from an explicitly named full-state checkpoint:

```bash
python -m ltx run \
  --config configs/native_cifar100_if100.yaml \
  --resume-method ddpm --resume-seed 0 \
  --resume-checkpoint /absolute/path/ckpt_250000.pt
```

Checkpoints are atomically replaced and the server retry policy caps their
interval, so SIGKILL recovery loses a bounded amount of work. Generated
Inception features are cached only while the sample path, size, and mtime still
match.

## Smoke and verification

The smoke config is infrastructure-only and deliberately bypasses the locked
paper preflight:

```bash
bash scripts/run.sh configs/smoke_ccua.yaml --skip-preflight
```

Before handing off a change:

```bash
python -m pytest -q
bash -n scripts/*.sh
git diff --check
```
