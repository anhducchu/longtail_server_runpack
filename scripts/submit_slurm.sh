#!/usr/bin/env bash
# Submit with: sbatch --array=0-8 scripts/submit_slurm.sh [config]
# Each array element owns one GPU and one method/seed run.
#SBATCH --job-name=longtail
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --signal=B:TERM@120
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/native_cifar100_if100.yaml}"
INDEX="${SLURM_ARRAY_TASK_ID:?submit this script with --array=0-8}"
if [[ -f "$ROOT/.env.local" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ROOT/.env.local"
  set +a
fi

VENV="${LTX_VENV:-$ROOT/.venv}"
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Missing $VENV; run scripts/bootstrap.sh before sbatch." >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"
exec python main.py run --config "$CONFIG" --index "$INDEX"
