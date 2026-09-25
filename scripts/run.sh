#!/usr/bin/env bash
# Prepare the CUDA environment, then run main.py directly.
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
  selected_python="$PYTHON_BIN"
elif command -v conda >/dev/null 2>&1; then
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "$conda_base/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda env update --file environment.yml --name "$ENV_NAME" --prune
  else
    conda env create --file environment.yml --name "$ENV_NAME"
  fi
  selected_python="$(conda run --no-capture-output -n "$ENV_NAME" which python)"
elif python -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  selected_python="$(command -v python)"
else
  echo "No CUDA-enabled Python or conda found." >&2
  exit 2
fi

export PYTHON_BIN="$selected_python"
bash scripts/bootstrap.sh
# shellcheck source=/dev/null
source "${LTX_VENV:-$ROOT/.venv}/bin/activate"

mapfile -t settings < <(python - "$CONFIG" <<'PY'
import sys
from experiment import load_experiment

experiment = load_experiment(sys.argv[1])
run = experiment.runs[0]
print("1" if any(item.evaluation.get("paper_metrics", False) for item in experiment.runs) else "0")
print(run.repo_dir)
print(run.dataset["root"])
print(run.metrics_root)
PY
)

if [[ "${settings[0]}" == "1" ]]; then
  python -m tools.prepare_cifar_metric_assets \
    --repo "${settings[1]}" \
    --data-root "${settings[2]}" \
    --output "${settings[3]}"
fi

exec python main.py run --config "$CONFIG" "$@"
