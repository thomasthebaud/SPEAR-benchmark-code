#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh
echo "data directory: $data_dir"

for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        for model in "${eval_models[@]}"; do
            echo "### Running base metrics for $model $split $subset ###"
            metadata=data/$protocol/outputs/$model/$split/$subset/metadata_verified.csv
            outputs=results/$protocol/$model/$split/$subset/base_metrics.csv
            $(python_cmd 'SB10' --gpu) bin/base_metrics.py \
                --metadata $metadata \
                --outputs $outputs \
                --asr-models "${asr_models[@]}" \
                --VAD-model "$VAD_model" \
                --UTMOS-model "$UTMOS_model" \
                --force-recompute 'WER' 'CER' 'interrupt' &
            sleep 1

        done
    done
done

wait 
exit