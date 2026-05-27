#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        for model in 'original' $llm_model; do
          metadata=data/$protocol/outputs/$model/$split/$subset/metadata.csv

          echo "Extract features for split $split subset $subset model $model"

          srun -p gpu --gpus 1 --job-name 'SB20' \
            python3 bin/naturalness/extract_features.py \
              --metadata $metadata \
              --min-len-question 3.0 &

        done
    done
done

wait
echo "All naturalness features extracted for original and $llm_model outputs."
exit
