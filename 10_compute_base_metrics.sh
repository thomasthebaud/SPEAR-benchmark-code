#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
echo "Transcribing outputs of model $llm_model using $asr"
echo "data directory: $data_dir"

for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        for model in $llm_model 'original'; do
            echo "### Running base metrics for $model $split $subset ###"
            metadata=data/$protocol/outputs/$model/$split/$subset/metadata.csv
            outputs=results/$protocol/$model/$split/$subset/base_metrics.csv
            srun -p cpu python3 bin/base_metrics.py \
                --metadata $metadata \
                --outputs $outputs

        done
    done
done

wait 
exit