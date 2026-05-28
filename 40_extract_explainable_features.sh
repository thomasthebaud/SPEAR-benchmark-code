#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh

for split in 'test' 'dev'; do
  for subset in 'improvised' 'naturalistic'; do
    for model in 'original' $llm_model; do
      metadata=data/$protocol/outputs/$model/$split/$subset/metadata.csv
      output=results/$protocol/$model/$split/$subset/distrib_baselines_features.csv

      echo "Extract explainable features for split:$split subset:$subset model:$model"
      $(python_cmd 'SB40' --cpu) bin/distrib_baselines/extract_features.py \
          --metadata "$metadata" \
          --relationships-csv "$seamless_assets_dir/relationships.csv" \
          --output "$output" &

    done
  done
done

wait

echo "all jobs finished!"

exit
