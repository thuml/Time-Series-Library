#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-timesnet}"
GPU="${GPU:-0}"
TASK="${1:-help}"
TARGET="${2:-all}"

task="$(printf '%s' "$TASK" | tr '[:upper:]' '[:lower:]')"
target="$(printf '%s' "$TARGET" | tr '[:upper:]' '[:lower:]')"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reproduce_timesnet.sh <task> [target]

Tasks:
  long_term       [all|etth1|etth2|ettm1|ettm2|ecl|traffic|weather|exchange|ili]
  short_term      [all]
  imputation      [all|etth1|etth2|ettm1|ettm2|ecl|weather]
  anomaly         [all|smd|msl|smap|swat|psm]
  classification  [all]
  all

Environment variables:
  CONDA_ENV   Conda environment name. Default: timesnet
  GPU         CUDA_VISIBLE_DEVICES value. Default: 0

Examples:
  bash scripts/reproduce_timesnet.sh long_term etth1
  bash scripts/reproduce_timesnet.sh imputation all
  bash scripts/reproduce_timesnet.sh short_term
  bash scripts/reproduce_timesnet.sh anomaly psm
  bash scripts/reproduce_timesnet.sh classification
  bash scripts/reproduce_timesnet.sh all
EOF
}

run_repo_script() {
  local script_path="$1"
  if [[ ! -f "$script_path" ]]; then
    echo "Script not found: $script_path" >&2
    exit 1
  fi

  echo
  echo "==> Running $script_path"
  echo "    conda env: $CONDA_ENV"
  echo "    CUDA_VISIBLE_DEVICES: $GPU"

  if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    {
      echo "set -euo pipefail"
      grep -v '^export CUDA_VISIBLE_DEVICES=' "$script_path"
    } | CUDA_VISIBLE_DEVICES="$GPU" bash -s
  else
    {
      echo "set -euo pipefail"
      grep -v '^export CUDA_VISIBLE_DEVICES=' "$script_path"
    } | CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n "$CONDA_ENV" bash -s
  fi
}

run_long_term() {
  case "$target" in
    all)
      run_repo_script "scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh"
      run_repo_script "scripts/long_term_forecast/ETT_script/TimesNet_ETTh2.sh"
      run_repo_script "scripts/long_term_forecast/ETT_script/TimesNet_ETTm1.sh"
      run_repo_script "scripts/long_term_forecast/ETT_script/TimesNet_ETTm2.sh"
      run_repo_script "scripts/long_term_forecast/ECL_script/TimesNet.sh"
      run_repo_script "scripts/long_term_forecast/Traffic_script/TimesNet.sh"
      run_repo_script "scripts/long_term_forecast/Weather_script/TimesNet.sh"
      run_repo_script "scripts/long_term_forecast/Exchange_script/TimesNet.sh"
      run_repo_script "scripts/long_term_forecast/ILI_script/TimesNet.sh"
      ;;
    etth1) run_repo_script "scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh" ;;
    etth2) run_repo_script "scripts/long_term_forecast/ETT_script/TimesNet_ETTh2.sh" ;;
    ettm1) run_repo_script "scripts/long_term_forecast/ETT_script/TimesNet_ETTm1.sh" ;;
    ettm2) run_repo_script "scripts/long_term_forecast/ETT_script/TimesNet_ETTm2.sh" ;;
    ecl|electricity) run_repo_script "scripts/long_term_forecast/ECL_script/TimesNet.sh" ;;
    traffic) run_repo_script "scripts/long_term_forecast/Traffic_script/TimesNet.sh" ;;
    weather) run_repo_script "scripts/long_term_forecast/Weather_script/TimesNet.sh" ;;
    exchange) run_repo_script "scripts/long_term_forecast/Exchange_script/TimesNet.sh" ;;
    ili) run_repo_script "scripts/long_term_forecast/ILI_script/TimesNet.sh" ;;
    *)
      echo "Unsupported long_term target: $TARGET" >&2
      usage
      exit 1
      ;;
  esac
}

run_short_term() {
  case "$target" in
    all) run_repo_script "scripts/short_term_forecast/TimesNet_M4.sh" ;;
    *)
      echo "Unsupported short_term target: $TARGET" >&2
      usage
      exit 1
      ;;
  esac
}

run_imputation() {
  case "$target" in
    all)
      run_repo_script "scripts/imputation/ETT_script/TimesNet_ETTh1.sh"
      run_repo_script "scripts/imputation/ETT_script/TimesNet_ETTh2.sh"
      run_repo_script "scripts/imputation/ETT_script/TimesNet_ETTm1.sh"
      run_repo_script "scripts/imputation/ETT_script/TimesNet_ETTm2.sh"
      run_repo_script "scripts/imputation/ECL_script/TimesNet.sh"
      run_repo_script "scripts/imputation/Weather_script/TimesNet.sh"
      ;;
    etth1) run_repo_script "scripts/imputation/ETT_script/TimesNet_ETTh1.sh" ;;
    etth2) run_repo_script "scripts/imputation/ETT_script/TimesNet_ETTh2.sh" ;;
    ettm1) run_repo_script "scripts/imputation/ETT_script/TimesNet_ETTm1.sh" ;;
    ettm2) run_repo_script "scripts/imputation/ETT_script/TimesNet_ETTm2.sh" ;;
    ecl|electricity) run_repo_script "scripts/imputation/ECL_script/TimesNet.sh" ;;
    weather) run_repo_script "scripts/imputation/Weather_script/TimesNet.sh" ;;
    *)
      echo "Unsupported imputation target: $TARGET" >&2
      usage
      exit 1
      ;;
  esac
}

run_anomaly() {
  case "$target" in
    all)
      run_repo_script "scripts/anomaly_detection/SMD/TimesNet.sh"
      run_repo_script "scripts/anomaly_detection/MSL/TimesNet.sh"
      run_repo_script "scripts/anomaly_detection/SMAP/TimesNet.sh"
      run_repo_script "scripts/anomaly_detection/SWAT/TimesNet.sh"
      run_repo_script "scripts/anomaly_detection/PSM/TimesNet.sh"
      ;;
    smd) run_repo_script "scripts/anomaly_detection/SMD/TimesNet.sh" ;;
    msl) run_repo_script "scripts/anomaly_detection/MSL/TimesNet.sh" ;;
    smap) run_repo_script "scripts/anomaly_detection/SMAP/TimesNet.sh" ;;
    swat) run_repo_script "scripts/anomaly_detection/SWAT/TimesNet.sh" ;;
    psm) run_repo_script "scripts/anomaly_detection/PSM/TimesNet.sh" ;;
    *)
      echo "Unsupported anomaly target: $TARGET" >&2
      usage
      exit 1
      ;;
  esac
}

run_classification() {
  case "$target" in
    all) run_repo_script "scripts/classification/TimesNet.sh" ;;
    *)
      echo "Unsupported classification target: $TARGET" >&2
      usage
      exit 1
      ;;
  esac
}

case "$task" in
  long_term|long-term)
    run_long_term
    ;;
  short_term|short-term)
    run_short_term
    ;;
  imputation)
    run_imputation
    ;;
  anomaly|anomaly_detection|anomaly-detection)
    run_anomaly
    ;;
  classification)
    run_classification
    ;;
  all)
    target="all"
    run_long_term
    run_short_term
    run_imputation
    run_classification
    run_anomaly
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unsupported task: $TASK" >&2
    usage
    exit 1
    ;;
esac
