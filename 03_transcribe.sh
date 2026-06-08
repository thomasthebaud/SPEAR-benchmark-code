#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh
echo "Transcribing outputs of model $llm_model"
echo "data directory: $data_dir"


for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        for model in "original";do #"${eval_models[@]}"; do
            for asr in 'Qwen3-ASR-0.6B' 'whisper-large-v3'; do
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
echo "All transcriptions completed for original and $llm_model outputs with both Qwen3-ASR-0.6B and whisper-large-v3 ASR models."
exit