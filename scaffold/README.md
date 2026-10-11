# Learned scaffold on a frozen DDPM

A pretrained class-conditional DDPM `θ` (CIFAR-100-LT) stays frozen. A small network `φ` — the scaffold — learns a per-pixel variance multiplier `λ_φ(x_t, t, c)` for a finite-step sampler of the DDIM family. The question the code is built to answer: does a state-dependent `λ_φ` beat the best fixed, time-only and shuffled-state alternatives at equal NFE?

## Method

One reverse edge `t → s` (`ᾱ_t < ᾱ_s`), with `ε_θ` the frozen, CFG-guided noise prediction:

```
x̂0  = (x_t − √(1−ᾱ_t) ε_θ) / √ᾱ_t
β̃   = (1−ᾱ_s)/(1−ᾱ_t) · (1 − ᾱ_t/ᾱ_s)              respaced posterior variance
σ²  = min(λ · β̃, (1 − 1e-4)(1 − ᾱ_s))               variance budget
x_s = √ᾱ_s x̂0 + √(1 − ᾱ_s − σ²) ε_θ + σ z
```

| `λ` | Sampler |
| --- | --- |
| `0` | deterministic DDIM |
| `1` | stochastic DDIM, `η = 1` (mean equals the DDPM posterior mean) |
| `λ_φ(x_t, t, c)` | learned scaffold |

The terminal edge (`s` = clean, `ᾱ_s = 1`) has `β̃ = 0`: it returns `x̂0` and injects no noise.

`φ` is trained with the variational bound of this K-step sampler. Each term is a closed-form KL between the forward posterior `q(x_s | x_t, x0)` and the sampler's own Gaussian `p_φ(x_s | x_t)`. The terminal edge is deterministic, so its reconstruction term (a discretized-Gaussian decoder around `x̂0`) is constant in `φ`; it is reported, not trained.

### What this sampler is, and is not

| | Learned stochastic DDIM (this code) | Marginal-preserving SDE |
| --- | --- | --- |
| Object | a K-step Markov chain; each edge is the Gaussian above | a continuous-time reverse SDE with diffusion `A_φ A_φᵀ` |
| Role of `λ_φ` | variance of one finite edge, with the mean adjusted DDIM-style | diffusion coefficient; needs the drift correction `½ ∇·(A Aᵀ)` when it depends on `x` |
| Marginals | preserved only for constant `λ` and an exact score, and then only as K → ∞ | preserved in continuous time for any admissible `A_φ` and an exact score |
| Status here | implemented, tested, frozen as `ddim_scaffold_v1` | not implemented |

`A_φ(x, t, c)` is diagonal: `A_φ = diag(√(λ_φ β̃))`. Because `λ_φ` enters a finite-step kernel without a divergence correction, a state-dependent `λ_φ` can shift the endpoint distribution even with an exact score. No claim in this repository relies on marginal preservation; any such claim needs its own measurement.

The variance budget makes the nominal `λ` misleading above an edge's threshold `(1−ᾱ_t)/(1−ᾱ_t/ᾱ_s)`. On the default 20-NFE `leading` grid the threshold ranges from 1.003 (edge 50→0) to 2.55 (edge 350→300), so `λ = 1.5` is clipped on 2 of 19 stochastic edges and `λ = 2` on 8. Every result therefore records the requested and the effective multiplier.

### Objective: `respaced_vlb` is a reference, not the test of adaptivity

- Its KL terms favour `λ ≥ 1`: for a given mean the optimal variance is `β̃` plus the squared mean error, and in every control tested no edge preferred `λ < 1`. It does not reward a less stochastic sampler, so it cannot show a benefit that needs `λ < 1`.
- On the default grid the last noisy edge 50→0 has `β̃ ≈ 1e-4` and dominates the bound, while its `λ` is clipped at 1.003. `diagnose` reports this per edge for the real score.
- Capped coordinates receive exactly zero gradient.

Training with it is possible and useful as a matched reference, and it stays available under `loss.kind = "respaced_vlb"`. An objective meant to demonstrate adaptive gains is added with `register_objective` in `loss.py`; the kernel does not change. Do not launch a long VLB run expecting it to answer the adaptivity question.

## Layout

| File | Contents |
| --- | --- |
| `sampling.py` | `Schedule`, timestep grids, `scaffold_transition`, `ddpm_transition`, the kernels and the single `sample` loop |
| `backbone.py` | frozen denoiser (`load_frozen_score`, `FrozenEpsilon`); learned scaffolds (`MLPScaffold`, `UNetScaffold`); controls (`FixedScaffold`, `TimeOnlyScaffold`, `ShuffledStateScaffold`) |
| `loss.py` | objective interface and registry, `RespacedVLB` with per-edge diagnostics |
| `main.py` | `preflight`, `diagnose`, `train` and `evaluate` commands |
| `metrics.py` | FID / IS (pinned InceptionV3), precision / recall (VGG-16 fc2, k = 3), optional classifier accuracy |
| `utils.py` | config, paths, atomic I/O, hashing, seeding, CIFAR-100-LT data |
| `config.json` | default experiment: MLP scaffold, 20 NFE |

Two interfaces keep the pieces independent:

- **Kernel** — `kernel(x_t, ε, t, s, labels) → Transition(mean, variance)`. `sample` draws from it and the loss scores it, so training and inference share one definition of the update.
- **Scaffold** — `scaffold(x_t, time, labels) → λ`. Any module with this signature plugs into `ScaffoldKernel`; changing the architecture or the loss does not touch `sampling.py`.
- **Objective** — `objective(epsilon_fn, kernel, x0, labels, generator=…) → (loss, details)`, built by `build_loss` from `loss.kind`. It reaches the sampler only through the kernel.

Timestep convention: indices `0 … T−1` are noisy levels, `CLEAN = −1` is the data endpoint. A grid with `K+1` nodes costs exactly `K` evaluations of `θ`.

## Configuration

`config.json` describes one experiment. `paths`, `dataset`, `diffusion`, `model` fix the frozen problem; `sampler` (grid and CFG) is shared by training and evaluation; `scaffold`, `loss`, `training`, `validation` define a trained `φ`; `evaluation` defines a panel.

A checkpoint's identity is everything except `evaluation`, so seeds, sample counts and arms can change without invalidating a trained scaffold. Any entry can be overridden on the command line:

```bash
--set sampler.grid.nfe=50
--set sampler.grid='{"stride": 10}'        # vendor DDIM grid 990, 980, …, 0
--set scaffold.inputs=time_class --set training.name=time_only_nfe20
```

Grids: `{"nfe": K, "spacing": "leading"}` evaluates at `0, T/K, …` and takes `x̂0` from index 0; `"trailing"` evaluates at `T−1, T−1−T/K, …`; `{"stride": n}` reproduces the vendor DDIM grid, which stops at index 0 without a clean edge.

## Deployment

Run from the repository root. Each step is a gate for the next: reproduce the baselines before judging the scaffold, and stop at the first failure.

```bash
# 0. Code-level checks (CPU; no checkpoint needed)
python -m pytest tests_independent -q
python -m pytest tests -q

# 1. Checks against the pinned data, reference statistics and pretrained weights
python -m pytest tests_independent --real-assets -q

# 2. Gate D: one command, writes scaffold/outputs/preflight/preflight.json, exit code 1 on failure
python -m scaffold.main preflight

# 3. Where does the objective put its loss and its gradient, with the real score?
python -m scaffold.main diagnose

# 4. Fixed-lambda sweep and reference samplers (no training)
python -m scaffold.main evaluate --arms ddim fixed_0.25 fixed_0.5 ddim_eta1 fixed_1.5 fixed_2.0 ddpm_fixedlarge

# 5. Only after 2 passes and the objective is settled: learned scaffold and time-only control
python -m scaffold.main train
python -m scaffold.main train --set scaffold.inputs=time_class --set training.name=time_only_nfe20
python -m scaffold.main evaluate
```

`train --resume` continues from `latest.pt` and refuses a changed config, score, data or kernel. `evaluate` always resumes, and a partial run (`--arms`) extends `summary.json` instead of replacing it. Independent training seeds are separate runs (`--set training.seed=… --set training.name=…`); evaluation seeds are `evaluation.seeds`.

### Preflight checks

| Check | Passes when |
| --- | --- |
| `score_checkpoint` | the checkpoint has the configured SHA-256 and loads strictly into the configured UNet |
| `noise_schedule` | `ᾱ` equals the vendor sampler's buffer bit for bit |
| `dataset_split` | class counts equal the long-tailed profile; pixels map to `[−1, 1]` |
| `score_noise_prediction` | at four noise levels θ is better at the true timestep than 15 % of the schedule away; up to 0.6 T it beats the trivial predictor, and near pure noise (where that predictor is near-optimal) it stays within a factor 10 of it |
| `ddim_parity` | `λ = 0` on the vendor grid, at CFG 0, the configured scale and 1.5: each edge reproduces the vendor's next state to 1e-5 (per step), and the endpoint gap is at most 10× the measured effect of a 1e-6 input nudge |
| `ddpm_parity` | the last 200 ancestral steps match the vendor `fixedlarge` and `fixedsmall` chains with shared noise |
| `metric_assets` | every reference file matches the manifest |
| `inception_preprocessing` | our Inception pipeline on real CIFAR-100 images reproduces the stored reference features |
| `baseline_panel` | DDIM, η=1 and DDPM arms sample and score to finite metrics (`--num-images`, default 1000) |

Parity is checked only between equivalent solvers on the same grid. The baseline panel's numbers are recorded, not thresholded: 1000 images give a rough FID, useful for spotting a broken pipeline, not for comparing arms.

### Outputs

```
scaffold/outputs/preflight/            preflight.json  panel/
scaffold/outputs/<training.name>_diagnostics/   initial.json  update_<n>_<weights>.json
scaffold/outputs/<training.name>/      config.json  reference_lambda1.json  train.jsonl  latest.pt  snapshot_<n>.pt  COMPLETE.train.json
scaffold/outputs/<evaluation.name>/    PROTOCOL.json  summary.json
    seed_<s>/<arm>/                    ARM.json  batches/  variance/  variance.json  samples.npy  metrics.json
```

- **Diagnostics** — one row per edge: `edge_kl_nats_per_dim`, `edge_gradient_norm`, `lambda_requested_mean`, `lambda_effective_mean`, `variance_clip_fraction`, `edge_weight`, `is_terminal`, the edge's `lambda_budget`, and the KL of the same states under fixed multipliers. The summary separates the full bound from the part that is trained and the part that has gradient, and names the dominant edge, zero-gradient edges and edges that prefer `λ < 1`.
- **Training** — `train.jsonl` has one row per update; checkpoint rows add the bound on two fixed panels (long-tailed training images and held-out CIFAR-100 test images), overall and per group. `reference_lambda1.json` is the same bound for `λ = 1`. Checkpoints record the package version and source hashes.
- **Evaluation** — `variance.json` gives, per edge, the requested and effective multiplier and the clipped fraction that the arm actually used. `summary.json` covers every scored cell on disk for the protocol: per-seed metrics, mean and standard deviation across seeds, and `paired_differences` against `evaluation.reference_arm` (DDIM) and against the best fixed-`λ` arm.

### Reading the numbers

- Paired differences are per evaluation seed (arms of a seed share all noise). The seed is the unit of replication; FID has no per-image difference. With three seeds the standard deviation is an indication, not a test.
- The best fixed arm is selected on the reported seeds, so comparisons against it favour it; confirm on fresh seeds.
- Recall is a global set-coverage statistic, not class coverage. Few-group FID at 5k samples (about 1700 images) is biased upwards and only comparable between arms of the same protocol.
- `ClassAccuracy` appears only when `evaluation.classifier` names a TorchScript CIFAR-100 classifier (`torchscript`, `sha256`, `mean`, `std`). None ships with this repository; validate one separately before using it.

### Evaluation arms

| `kind` | Fields | Meaning |
| --- | --- | --- |
| `fixed` | `lambda` | constant multiplier (`0` = DDIM, `1` = η=1); the default config sweeps 0, 0.25, 0.5, 1, 1.5, 2 |
| `ddpm` | `variance_type` | ancestral DDPM, `fixedsmall` or `fixedlarge` |
| `learned` | `checkpoint`, `weights` (`ema`/`online`), `state` (`own`/`shuffled`) | trained scaffold; `shuffled` gives each sample the state of another sample of the same class |

All arms of a seed share labels, initial latents and Brownian draws. The shuffled-state arm needs at least two samples of every class per batch, which is why `evaluation.batch_size` is 200 for 100 classes. A learned arm whose checkpoint was trained on another grid or CFG is refused unless it sets `"allow_protocol_mismatch": true`; another score, schedule or kernel version is always refused. Changing the protocol requires a new `evaluation.name`.

## Verified properties

- `λ = 0` reproduces the vendor DDIM sampler on its grid; `DDPMKernel` reproduces the vendor ancestral chain for `fixedlarge` and `fixedsmall` with shared noise (max abs error < 2e-4).
- With an exact Gaussian score the endpoint variance of a fixed-`λ` sampler matches its closed-form finite-step value at 20 / 50 / 100 NFE, and the error shrinks monotonically with NFE.
- Seeds are independent, runs are resumable bit-for-bit, and lost shards are regenerated identically.

On that Gaussian control, at 20 NFE with `leading` spacing, the ratio of endpoint to data standard deviation for directions of data std 0.5 / 0.2 / 0.05 is 0.86 / 0.73 / 0.44 for `λ = 0` and 0.78 / 0.64 / 0.33 for `λ = 1`: within this family a larger `λ` contracts more. An improvement of a learned `λ` over DDIM on real data therefore has to come from correcting score error, not finite-step error.

## Status of v1 (`1.0.0rc2`, kernel `ddim_scaffold_v1`)

- **Verified on CPU with synthetic and stub models:** kernel and loss maths against independent numpy oracles, vendor parity, reproducibility, diagnostics, summary merging, paired statistics, preflight logic.
- **Verified with real assets:** data split, noise schedule, metric assets and the Inception preprocessing.
- **Not yet verified:** anything involving the pretrained score — `preflight` checks `score_checkpoint`, `score_noise_prediction`, `ddim_parity`, `ddpm_parity`, `baseline_panel`, and the four tests in `tests_independent/test_indep_pretrained.py`.
- **Open:** an objective able to demonstrate adaptive gains; a validated CIFAR-100 classifier for class accuracy.

The earlier Euler–Maruyama ELBO pipeline was removed; its sources are archived in `scaffold/outputs/legacy_source_20261011.tar.gz`.
