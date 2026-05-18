#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh

echo "Train dev-set explainable-feature baselines and score test utterances for model:$llm_model"

srun -p cpu \
  python3 bin/distrib_baselines/compute_results.py \
    --results-root "results/$protocol" \
    --model "$llm_model" \
    --subsets improvised naturalistic

echo "all jobs finished!"

exit
