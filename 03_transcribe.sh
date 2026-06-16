#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh
echo "Transcribing outputs of model $llm_model"
echo "data directory: $data_dir"


for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        for model in "${eval_models[@]}"; do
            for asr in "${asr_models[@]}"; do
                echo "running analysis for $model $split $subset ASR model=$asr"
                data_dir="data/$protocol/outputs/$model"
                $(python_cmd 'SB03' --gpu) bin/transcribe.py \
                    --data_dir "$data_dir" \
                    --model "$asr" \
                    --split "$split" \
                    --subset "$subset" \
                    --output_dir "data/$protocol/outputs/$model/$split/$subset" &
            done
        done
    done
done

wait
echo "All transcriptions completed for original and $llm_model outputs with ${asr_models[@]} ASR models."
exit