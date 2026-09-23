#!/usr/bin/env bash
# Main CCUA-backbone CIFAR-100-LT IF100 campaign: DDPM/CBDM/CCUA x 3 seeds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env.local" ]]; then
  set -a; source "$ROOT/.env.local"; set +a
elif [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
fi

CAMPAIGN_NAME="ccua_cifar100_if100_v1"
# Viettel's project home is full while the attached data volume has room for
# checkpoints/samples.  Respect an explicit override; otherwise use that
# volume when it exists, and fall back to the local runpack directory on other
# machines.
if [[ -z "${LTX_RUNS_ROOT:-}" && -d "/home/nvidia-lab/data_mount" ]]; then
  export LTX_RUNS_ROOT="/home/nvidia-lab/data_mount/longtail_server_runpack/runs"
fi
RUNS_ROOT="${LTX_RUNS_ROOT:-$ROOT/runs}"
LOG_DIR="$RUNS_ROOT/$CAMPAIGN_NAME/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_$(date -u +%Y%m%dT%H%M%SZ).log"
ln -sfn "$RUN_LOG" "$RUNS_ROOT/$CAMPAIGN_NAME/latest.log"
exec > >(tee -a "$RUN_LOG") 2>&1
echo "[run] native campaign log: $RUN_LOG"
echo "[run] runs root: $RUNS_ROOT"

bash scripts/bootstrap.sh
source "${LTX_VENV:-$ROOT/.venv}/bin/activate"

if [[ "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_MODE=online but WANDB_API_KEY is empty; set it or use WANDB_MODE=offline/disabled." >&2
  exit 2
fi
if [[ "${WANDB_MODE:-online}" == "disabled" ]]; then
  echo "[run] warning: W&B is disabled; results remain local under $RUNS_ROOT" >&2
fi

REPOS_ROOT="${LTX_REPOS_ROOT:-$ROOT/third_party}"
DATA_ROOT="${LTX_DATA_ROOT:-$ROOT/data}"
METRICS_ROOT="${LTX_METRICS_ROOT:-$REPOS_ROOT/CCUA-DDPM/stats}"
python tools/prepare_cifar_metric_assets.py \
  --repo "$REPOS_ROOT/CCUA-DDPM" \
  --data-root "$DATA_ROOT" \
  --output "$METRICS_ROOT" \
  --datasets cifar100

# The main baseline now uses the CCUA-DDPM U-Net for all four objectives.
# Each CCUA-DDPM baseline starts or resumes only in its own native run
# directory. Checkpoints from retired hosts remain audit artifacts and are not
# loaded into this campaign.

RUN_ARGS=(--config configs/native_cifar100_if100.yaml)
[[ -n "${LTX_GPU_IDS:-}" ]] && RUN_ARGS+=(--gpus "$LTX_GPU_IDS")
[[ -n "${LTX_TASKS_PER_GPU:-}" ]] && RUN_ARGS+=(--per-gpu "$LTX_TASKS_PER_GPU")
[[ -n "${LTX_MAX_CONCURRENT:-}" ]] && RUN_ARGS+=(--jobs "$LTX_MAX_CONCURRENT")
RUN_ARGS+=("$@")

set +e
python -m ltx.cli run "${RUN_ARGS[@]}"
campaign_status=$?
python tools/report_native_cifar.py --config configs/native_cifar100_if100.yaml --wandb
report_status=$?
set -e

echo "[run] campaign_status=$campaign_status report_status=$report_status"
echo "[run] native campaign log: $RUN_LOG"
[[ "$campaign_status" -eq 0 && "$report_status" -eq 0 ]]
