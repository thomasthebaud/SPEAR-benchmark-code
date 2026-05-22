#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source openai_keys.sh

echo "Running inference with model $llm_model on seamless interactions"
echo "Input data directory: $data_dir/inputs"
echo "Output data directory: $data_dir/outputs/$llm_model"

for split in 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        echo "on split $split and subset $subset"
        srun -p cpu python3 bin/run_LLM_inference.py \
            --audio_dir $data_dir/inputs \
            --output_dir $data_dir/outputs/$llm_model \
            --model $llm_model \
            --prompt "None" \
            --split $split \
            --subset $subset \
            --openai-api-key "$openai_api_key" \
            --org "$org" 

    done
    exit
done