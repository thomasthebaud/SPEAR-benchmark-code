#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh
echo "Transcribing outputs of model $llm_model using $asr"
echo "data directory: $data_dir"

for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        for model in "${eval_models[@]}"; do
            echo "### Running base metrics for $model $split $subset ###"
            metadata=data/$protocol/outputs/$model/$split/$subset/metadata.csv
            outputs=results/$protocol/$model/$split/$subset/base_metrics.csv
            $(python_cmd 'SB10' --gpu) bin/base_metrics.py \
                --metadata $metadata \
                --outputs $outputs &

        done
    done
done

wait 
exit