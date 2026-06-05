#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh
for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        for model in 'original' "${eval_models[@]}"; do
          metadata=data/$protocol/outputs/$model/$split/$subset/metadata.csv
          metrics=results/$protocol/$model/$split/$subset
          pickle_dir=data/$protocol/inputs/$split/$subset

          echo "Predict Naturalness score for split:$split subset:$subset model:$model"

          $(python_cmd 'SB22' --cpu) bin/naturalness/score.py \
              --metadata $metadata \
              --outputs $metrics \
              --pickle-dir $pickle_dir \
              --assets-dir $seamless_assets_dir \
              --model-path models/naturalness/last_model.pt &


        done
    done
done

wait
echo "All naturalness scores computed for original and ${eval_models[@]} outputs."