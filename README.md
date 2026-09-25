# Long-tail diffusion baselines

This repository runs DDPM, CBDM, T2H, and CCUA on one pinned CCUA-DDPM
implementation. All objectives share the CIFAR-100-LT dataset, U-Net,
optimizer, sampler, checkpoint format, and evaluator.

Archived research code remains on the local branch
`archive/research-wip-20260922`.

## Read the code

Start at `main.py`. The complete runtime flow is:

```text
main.py:run_one()
  -> experiment.py:train()
       -> third_party/CCUA-DDPM/main.py:train()
       -> third_party/CCUA-DDPM/diffusion.py:GaussianDiffusionTrainer.forward()
  -> experiment.py:sample()
       -> third_party/CCUA-DDPM/main.py:sample()
  -> experiment.py:evaluate()
       -> tools/evaluate_ccua.py
```

There is no internal scheduler, worker process, SQLite state database, or
adapter class. `main.py` runs selected experiments sequentially. Shell or
Slurm owns parallelism.

The objective switch reaches the same diffusion trainer:

| Method | Training objective |
| --- | --- |
| DDPM | DDPM noise-prediction loss |
| CBDM | DDPM + CBDM balancing loss |
| T2H | DDPM with the transferred target |
| CCUA | DDPM + alignment + unconditional contrastive loss |

## Repository map

| Path | Purpose |
| --- | --- |
| `main.py` | Public flow: select runs, then train → sample → evaluate |
| `experiment.py` | Flat config loader, checkpoint resume, and direct vendor commands |
| `metrics.py` | Shared paper metric implementations |
| `configs/` | Scientific parameters only |
| `third_party/CCUA-DDPM/` | Pinned model, objective, trainer, and sampler |
| `tools/` | Metric asset preparation, evaluation, and report generation |
| `scripts/` | Environment bootstrap, local launch, and Slurm array launch |
| `tests/` | Scientific-contract and metric regression tests |

Generated data, checkpoints, arrays, logs, reports, and W&B files remain local
and are ignored by Git.

## Configure

```bash
cp .env.example .env.local
# Fill local paths and optional W&B report settings.
```

The YAML files contain the scientific configuration. Local paths come from
`.env.local`; temporary run selection comes from CLI flags.

```bash
python main.py plan --config configs/native_cifar100_if100.yaml
python main.py preflight --config configs/native_cifar100_if100.yaml
python main.py run --help
```

The main config contains DDPM/CBDM/CCUA × seeds 0, 1, and 2. The seed-0 config
contains DDPM/CBDM/T2H.

## Run

Run the complete nine-run campaign sequentially:

```bash
bash scripts/run.sh
```

Run one experiment on one GPU:

```bash
bash scripts/run.sh \
  configs/native_cifar100_if100.yaml \
  --method ddpm --seed 0 --gpu 0
```

Run only one lifecycle stage:

```bash
python main.py run --method ddpm --seed 0 --stage train
python main.py run --method ddpm --seed 0 --stage sample
python main.py run --method ddpm --seed 0 --stage evaluate
```

Rerunning the same command skips completed outputs. Training resumes the newest
complete `ckpt_<step>.pt`; checkpoint replacement is atomic. An external
full-state checkpoint can be selected explicitly:

```bash
python main.py run --method ddpm --seed 0 \
  --resume-checkpoint /absolute/path/ckpt_250000.pt
```

## Multi-GPU and Slurm

`python main.py plan` prints a stable index for each method/seed. Launch one
process per GPU or use a Slurm array. Prepare the environment and shared metric
assets once before submitting the array; array jobs never mutate the shared
environment:

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
python -m tools.prepare_cifar_metric_assets \
  --repo third_party/CCUA-DDPM --data-root "${LTX_DATA_ROOT:-./data}" \
  --output "${LTX_METRICS_ROOT:-third_party/CCUA-DDPM/stats}"
sbatch --array=0-8 scripts/submit_slurm.sh
```

For the three-run seed-0 config:

```bash
sbatch --array=0-2 scripts/submit_slurm.sh \
  configs/native_cifar100_if100_seed0.yaml
```

Each array element runs exactly one index and therefore owns one GPU.

## Report and verify

```bash
python -m tools.report_native_cifar \
  --config configs/native_cifar100_if100.yaml --wandb

python -m pytest -q
bash -n scripts/*.sh
git diff --check
```
