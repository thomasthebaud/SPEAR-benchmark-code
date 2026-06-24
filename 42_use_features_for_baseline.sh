#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh

run_scores=0
run_clusters=0

if [[ $# -eq 0 ]]; then
  run_scores=1
  run_clusters=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scores|--stage1)
      run_scores=1
      ;;
    --clusters|--stage2)
      run_clusters=1
      ;;
    --all)
      run_scores=1
      run_clusters=1
      ;;
    -h|--help)
      cat <<EOF
Usage: bash 42_use_features_for_baseline.sh [--scores] [--clusters] [--all]

Stages:
  --scores    Stage 1: train dev-set explainable-feature baselines and score test utterances
  --clusters  Stage 2: analyze explainable-feature correlations and compute PCA cluster features
  --all       Run both stages

If no stage is passed, both stages are run.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$run_scores" -eq 1 ]]; then
  echo "Stage 1: train dev-set explainable-feature baselines and score test utterances for model:"${eval_models[@]}""
  for model in "${eval_models[@]}"; do
    $(python_cmd 'SB42-S1' --cpu) bin/distrib_baselines/compute_features_scores.py \
        --results-root "results/$protocol" \
        --model $model \
        --subsets improvised naturalistic \
        --ignore-features "${ignored_explainable_features[@]:-}" &
  done
  wait
  echo "All explainable-feature baseline scores computed for ${eval_models[@]} outputs."

fi

if [[ "$run_clusters" -eq 1 ]]; then
  echo "Stage 2: analyze explainable-feature correlations jointly for original and model:"${eval_models[@]}""
  for model in "${eval_models[@]}"; do
    $(python_cmd 'SB42-S2' --cpu) bin/distrib_baselines/analyse_correlations.py \
        --results-root "results/$protocol" \
        --model $model \
        --subsets improvised naturalistic \
        --thresholds 0.8 \
        --ignore-features "${ignored_explainable_features[@]:-}" &

  done
  wait
  echo "All explainable-feature correlations analyzed for ${eval_models[@]} outputs."

fi

exit
