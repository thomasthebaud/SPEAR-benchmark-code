#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
echo "Transcribing outputs of model $llm_model using $asr"
echo "data directory: $data_dir"

for split in 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        for model in 'original' $llm_model; do
            echo "running analysis for $model $split $subset"
            data_dir="data/$protocol/outputs/$model"
            srun -p gpu --gpus 1 python3 bin/transcribe.py \
                --data_dir $data_dir \
                --model $asr \
                --split $split \
                --subset $subset
        done
    done
    exit
done