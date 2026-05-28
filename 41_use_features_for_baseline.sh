#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh

echo "Train dev-set explainable-feature baselines and score test utterances for model:$llm_model"

# $(python_cmd 'SB41-S1' --cpu) bin/distrib_baselines/compute_results.py \
#     --results-root "results/$protocol" \
#     --model "$llm_model" \
#     --subsets improvised naturalistic

$(python_cmd 'SB41-S2' --cpu) bin/distrib_baselines/cluster_metrics.py \
    --results-root "results/$protocol" \
    --model "$llm_model" \
    --subsets improvised naturalistic

exit
