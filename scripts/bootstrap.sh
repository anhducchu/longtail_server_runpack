#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${LTX_VENV:-$ROOT/.venv}"
REPOS_ROOT="${LTX_REPOS_ROOT:-$ROOT/third_party}"
INSTALL_REPO_DEPS="${LTX_INSTALL_REPO_DEPS:-1}"

# Always converge the local venv to the interpreter selected by run_server.
# This avoids retaining a stale venv after the pinned conda environment changes.
"$PYTHON_BIN" -m venv --upgrade --system-site-packages "$VENV"
source "$VENV/bin/activate"
# Keep compatibility with the CUDA PyTorch builds commonly preinstalled on
# research servers (torch 2.12 currently requires setuptools < 82).
python -m pip install --upgrade "pip<26" "setuptools<82" wheel
python -m pip install -e .

if [[ ! -f "$REPOS_ROOT/THIRD_PARTY_MANIFEST.json" ]]; then
  echo "[bootstrap] missing third_party/THIRD_PARTY_MANIFEST.json; refusing an unverifiable vendor tree." >&2
  exit 2
fi

# CCUA-DDPM is the only active model/data/sampler backbone.  Metric assets are
# also stored below this tree, so the active runner has no dependency on the
# any retired standalone source directories.
CCUA_ROOT="$REPOS_ROOT/CCUA-DDPM"
if [[ ! -d "$CCUA_ROOT" || ! -f "$CCUA_ROOT/main.py" || ! -f "$CCUA_ROOT/model/model.py" ]]; then
  echo "[bootstrap] missing active CCUA-DDPM source at $CCUA_ROOT." >&2
  exit 2
fi
python third_party/patches/apply_ccua_imagenet_lt.py "$CCUA_ROOT"
python third_party/patches/apply_ccua_sample_export.py "$CCUA_ROOT"
mkdir -p "$CCUA_ROOT/stats"
echo "[bootstrap] CCUA-DDPM-only backbone ready; legacy source trees are not required"

if [[ "$INSTALL_REPO_DEPS" == "1" ]]; then
  # Preserve the server's CUDA/PyTorch installation.  The 2021 CBDM source
  # lists obsolete TensorFlow/PyTorch pins which are not used by this runner.
  python -m pip install -r "$ROOT/requirements.runpack.txt"
fi

# Authenticate once so every worker and the final report share one identity
# instead of relying on ambient env vars. Never echo the key itself.
if [[ -n "${WANDB_API_KEY:-}" && "${WANDB_MODE:-online}" != "offline" ]]; then
  if wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1; then
    echo "[bootstrap] W&B authenticated as ${WANDB_ENTITY:-default entity}"
  else
    echo "[bootstrap] W&B login failed; online workers will stop instead of silently losing logs" >&2
  fi
fi

{
  date -Is
  sha256sum "$REPOS_ROOT/THIRD_PARTY_MANIFEST.json"
  python - <<'PY'
import torch
print('python_torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
PY
} > "$REPOS_ROOT/VENDOR_AND_ENV.txt"

mkdir -p "${LTX_RUNS_ROOT:-$ROOT/runs}" "${LTX_DATA_ROOT:-$ROOT/data}" "$ROOT/wandb"

echo
printf 'Bootstrap complete. The runner reads this runpack\047s .env.local (or $LTX_ENV_FILE).\n  source %q\n  python -m ltx.cli preflight --config configs/native_cifar100_if100.yaml\n  bash scripts/run_server_c100.sh\n' "$VENV/bin/activate"
