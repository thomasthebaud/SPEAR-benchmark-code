#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh


for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        for model in "${eval_models[@]}"; do
          metadata=data/$protocol/outputs/$model/$split/$subset/metadata.csv
          metrics=results/$protocol/$model/$split/$subset
          pickle_dir=data/$protocol/inputs/$split/$subset

          echo "Predict Naturalness score for split:$split subset:$subset model:$model"

          $(python_cmd 'SB22-S1' --gpu) bin/naturalness/score.py \
              --metadata $metadata \
              --outputs $metrics \
              --pickle-dir $pickle_dir \
              --assets-dir $seamless_assets_dir \
              --model-path $emo_naturalness_checkpoint &


        done
    done
done

wait
echo "All naturalness scores computed for original and ${eval_models[@]} outputs."


for subset in 'improvised' 'naturalistic'; do
    ref_dir=results/$protocol/original/dev/$subset
    for model in "${eval_models[@]}"; do
        data_dir=results/$protocol/$model/test/$subset

        echo "Normalize Naturalness score for subset:$subset model:$model using ref_dir:$ref_dir"

        $(python_cmd 'SB22-S2' --cpu) bin/naturalness/normalize_score.py \
            --ref-dir "$ref_dir" \
            --data-dir "$data_dir" &
    done
done

wait
echo "All naturalness scores normalized for original and ${eval_models[@]} outputs."
exit
