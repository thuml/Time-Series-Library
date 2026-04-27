#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-timesnet}"
GPU="${GPU:-0}"
LOG_DIR="${1:-$ROOT_DIR/reproduction_logs/timesnet_$(date -u +%Y%m%dT%H%M%SZ)}"
CONDA_SETUP="${CONDA_SETUP:-/root/miniconda3/etc/profile.d/conda.sh}"

mkdir -p "$LOG_DIR"
mkdir -p "$ROOT_DIR/reproduction_logs"
ln -sfn "$LOG_DIR" "$ROOT_DIR/reproduction_logs/latest"

if [[ ! -f "$CONDA_SETUP" ]]; then
  echo "Conda setup script not found: $CONDA_SETUP" >&2
  exit 1
fi

source "$CONDA_SETUP"
conda activate "$CONDA_ENV"

write_metadata() {
  {
    echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root_dir=$ROOT_DIR"
    echo "conda_env=$CONDA_ENV"
    echo "gpu=$GPU"
    echo "hostname=$(hostname)"
    echo
    echo "[python]"
    python --version
    echo
    echo "[torch]"
    python - <<'PY'
import torch
print('torch_version=', torch.__version__)
print('cuda_available=', torch.cuda.is_available())
print('device_count=', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device0=', torch.cuda.get_device_name(0))
PY
    echo
    echo "[nvidia-smi]"
    nvidia-smi
  } > "$LOG_DIR/metadata.txt"
}

refresh_summary() {
  python scripts/summarize_timesnet_results.py \
    --output "$LOG_DIR/TIMESNET_REPRODUCTION_RESULTS.md"
  python scripts/summarize_timesnet_results.py \
    --output "$ROOT_DIR/TIMESNET_REPRODUCTION_RESULTS.md"
}

run_step() {
  local step_name="$1"
  shift
  local step_log="$LOG_DIR/${step_name}.log"

  {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START $step_name"
    "$@"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] END $step_name"
  } 2>&1 | tee "$step_log"

  refresh_summary | tee -a "$step_log"
}

write_metadata
refresh_summary > "$LOG_DIR/summary_refresh.log" 2>&1 || true

run_step 01_long_term env CONDA_ENV="$CONDA_ENV" GPU="$GPU" bash scripts/reproduce_timesnet.sh long_term all
run_step 02_short_term env CONDA_ENV="$CONDA_ENV" GPU="$GPU" bash scripts/reproduce_timesnet.sh short_term
run_step 03_imputation env CONDA_ENV="$CONDA_ENV" GPU="$GPU" bash scripts/reproduce_timesnet.sh imputation all
run_step 04_classification env CONDA_ENV="$CONDA_ENV" GPU="$GPU" bash scripts/reproduce_timesnet.sh classification
run_step 05_anomaly env CONDA_ENV="$CONDA_ENV" GPU="$GPU" bash scripts/reproduce_timesnet.sh anomaly all

refresh_summary > "$LOG_DIR/summary_refresh.log" 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ > "$LOG_DIR/completed_at.txt"
