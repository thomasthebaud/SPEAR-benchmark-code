#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        for model in 'original' $llm_model; do
          metadata=data/$protocol/outputs/$model/$split/$subset/${split}_${subset}_metadata.csv

          echo "Extract features for split $split subset $subset model $model"

          srun -p gpu --gpus 1 \
            python3 bin/naturalness/extract_features.py \
              --metadata $metadata

        done
    done
done

  
