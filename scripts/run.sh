#!/usr/bin/env bash
# Bootstrap and launch one campaign. The first positional argument may be a
# config path; remaining arguments are passed to `ltx run`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env.local" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ROOT/.env.local"
  set +a
fi

CONFIG="configs/native_cifar100_if100.yaml"
if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG="$1"
  shift
fi

ENV_NAME="${LTX_CONDA_ENV:-longtail-ccua}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  candidate="$PYTHON_BIN"
elif command -v conda >/dev/null 2>&1; then
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "$conda_base/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda env update --file environment.yml --name "$ENV_NAME" --prune
  else
    conda env create --file environment.yml --name "$ENV_NAME"
  fi
  candidate="$(conda run --no-capture-output -n "$ENV_NAME" which python)"
elif command -v python >/dev/null 2>&1 \
  && python -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  candidate="$(command -v python)"
else
  echo "No CUDA-enabled Python or conda found. Install conda, then rerun." >&2
  exit 2
fi

export PYTHON_BIN="$candidate"
if [[ -z "${LTX_RUNS_ROOT:-}" && -d /home/nvidia-lab/data_mount ]]; then
  export LTX_RUNS_ROOT=/home/nvidia-lab/data_mount/longtail_server_runpack/runs
fi

bash scripts/bootstrap.sh
# shellcheck source=/dev/null
source "${LTX_VENV:-$ROOT/.venv}/bin/activate"

read -r campaign_name protocol < <(
  python - "$CONFIG" <<'PY'
import sys
from ltx.config import load_campaign

campaign = load_campaign(sys.argv[1]).raw.get("campaign", {})
print(campaign.get("name", "campaign"), campaign.get("protocol", "none"))
PY
)

runs_root="${LTX_RUNS_ROOT:-$ROOT/runs}"
log_dir="$runs_root/$campaign_name/logs"
mkdir -p "$log_dir"
run_log="$log_dir/run_$(date -u +%Y%m%dT%H%M%SZ).log"
ln -sfn "$run_log" "$runs_root/$campaign_name/latest.log"
exec > >(tee -a "$run_log") 2>&1
echo "[run] campaign=$campaign_name config=$CONFIG"
echo "[run] runs_root=$runs_root log=$run_log"

if [[ "$protocol" == native_cifar_v1 ]]; then
  repos_root="${LTX_REPOS_ROOT:-$ROOT/third_party}"
  python tools/prepare_cifar_metric_assets.py \
    --repo "$repos_root/CCUA-DDPM" \
    --data-root "${LTX_DATA_ROOT:-$ROOT/data}" \
    --output "${LTX_METRICS_ROOT:-$repos_root/CCUA-DDPM/stats}" \
    --datasets cifar100
fi

run_args=(--config "$CONFIG")
[[ -n "${LTX_GPU_IDS:-}" ]] && run_args+=(--gpus "$LTX_GPU_IDS")
[[ -n "${LTX_TASKS_PER_GPU:-}" ]] && run_args+=(--per-gpu "$LTX_TASKS_PER_GPU")
[[ -n "${LTX_MAX_CONCURRENT:-}" ]] && run_args+=(--jobs "$LTX_MAX_CONCURRENT")
run_args+=("$@")

set +e
python -m ltx run "${run_args[@]}"
campaign_status=$?
report_status=0
if [[ "$protocol" == native_cifar_v1 ]]; then
  python tools/report_native_cifar.py --config "$CONFIG" --wandb
  report_status=$?
fi
set -e

echo "[run] campaign_status=$campaign_status report_status=$report_status"
[[ "$campaign_status" -eq 0 && "$report_status" -eq 0 ]]
