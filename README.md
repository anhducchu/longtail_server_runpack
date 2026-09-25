# Long-tail diffusion baseline runpack

This repository runs DDPM, CBDM, T2H, and CCUA through one pinned
`third_party/CCUA-DDPM/` host. The four objectives share the CIFAR-100-LT
split, U-Net, optimizer, checkpoint format, sampler, and evaluator. Any other
objective or runtime host is rejected before launch.

Archived research code is preserved on the local branch
`archive/research-wip-20260922`; it is not part of the `main` runtime.

## Start here

- To understand the repo, open `main.py` and start at its `main()` function.
- To run the full server workflow, use `scripts/run.sh`.
- Scientific parameters live in `configs/native_cifar100_if100.yaml`; machine
  and retry parameters live in `configs/server.yaml`.

```text
main.py:main()
  -> main.py:run_campaign()                    load config + preflight
  -> runner/scheduler.py:Scheduler.run()       choose GPU and launch tasks
  -> runner/worker.py:main()                   execute one task safely
  -> runner/adapter.py:CCUAAdapter.phases()    train -> sample -> metrics
  -> third_party/CCUA-DDPM/main.py:train()     data, U-Net, optimizer, loop
  -> third_party/CCUA-DDPM/diffusion.py:
       GaussianDiffusionTrainer.forward()      DDPM/CBDM/T2H/CCUA objective
```

There is no installable framework layer. `main.py` is the CLI; `runner/`
contains only the infrastructure needed to execute and recover experiments.

The important handoff is in `CCUAAdapter._train_phase()`: it translates one
resolved task into the arguments for the vendored trainer. In that trainer,
`train()` constructs `GaussianDiffusionTrainer`; its `forward()` computes the
four loss components, and the training loop combines the components enabled by
the chosen objective. Sampling follows the same path through
`CCUAAdapter._sample_phase()` to `third_party/CCUA-DDPM/main.py:sample()`.

| Objective | Adapter enables | Training objective |
| --- | --- | --- |
| DDPM | no additional flag | DDPM noise-prediction loss |
| CBDM | `cbdm`, `cb_tau` | DDPM + CBDM balancing loss |
| T2H | `transfer_x0` | DDPM loss with the transferred target |
| CCUA | `ccua_al`, `ccua_ucl` | DDPM + alignment + unconditional contrastive loss |

### Where parameters come from

| Parameter type | Source | Examples |
| --- | --- | --- |
| Scientific contract | `configs/native_cifar100_if100.yaml` | objective, seeds, batch size, learning rate, diffusion steps, sampling |
| Machine policy | `configs/server.yaml` | GPUs, concurrency, storage roots, retry, checkpoint interval |
| One-launch override | `python main.py run --help` | selected GPUs, job limit, explicit resume checkpoint |
| Local environment | `.env.local` | paths, W&B credentials and mode |

`runner/config.py:load_campaign()` merges those files into concrete `Task`
objects. From that point onward, scheduler, worker, and adapter receive resolved
values; they do not invent another scientific configuration layer.

## Repository map

| Path | Purpose |
| --- | --- |
| `main.py` | Single CLI entrypoint: plan, preflight, run, status, retry, stop |
| `configs/` | Main 9-task campaign, seed-0 T2H comparison, smoke, and server policy |
| `runner/` | Config, preflight, scheduling, worker, state, adapter, and metrics |
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
python main.py plan --config configs/native_cifar100_if100.yaml
python main.py preflight --config configs/native_cifar100_if100.yaml
```

Run the main campaign or the seed-0 T2H comparison:

```bash
bash scripts/run.sh
bash scripts/run.sh configs/native_cifar100_if100_seed0.yaml
```

`scripts/run.sh` selects or creates the CUDA environment, bootstraps the local
venv, prepares missing CIFAR metric assets, launches the scheduler, and writes
the report. CLI flags after the optional config path pass through to
`main.py run`:

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
python main.py status --config configs/native_cifar100_if100.yaml
python main.py status --watch 20 --config configs/native_cifar100_if100.yaml
python main.py retry-failed --config configs/native_cifar100_if100.yaml
python main.py stop --config configs/native_cifar100_if100.yaml
```

Rerunning the same campaign resumes from the newest complete local checkpoint.
To resume one row from an explicitly named full-state checkpoint:

```bash
python main.py run \
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
