#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
for split in 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        for model in 'original' $llm_model; do
          metadata=data/$protocol/outputs/$model/$split/$subset/${split}_${subset}_metadata.csv
          metrics=results/$protocol/$model/$split/$subset
          pickle_dir=data/$protocol/inputs/$split/$subset

          echo "Predict Naturalness score for split:$split subset:$subset model:$model"

          srun -p cpu \
            python3 bin/naturalness/score.py \
              --metadata $metadata \
              --outputs $metrics \
              --pickle-dir $pickle_dir \
              --assets-dir $seamless_assets_dir


        done
    done
done

  
