#!/usr/bin/env bash
#SBATCH --job-name=longtail-deadline
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@120
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec bash scripts/run.sh "$@"
