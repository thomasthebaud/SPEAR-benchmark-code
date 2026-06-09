#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source openai_keys.sh
source cmd.sh

echo "Running inference with model $llm_model on seamless interactions"
echo "Input data directory: $data_dir/inputs"
echo "Output data directory: $data_dir/outputs/$llm_model"
prompt="You are participating in a natural spoken conversation.\
    Answer when it feels natural, not only at the very end.\
    Keep responses conversational and concise.\
    If the user interrupts, stop and respond to the latest user speech."

for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        echo "on split $split and subset $subset"
        $(python_cmd 'SB02' --gpu) bin/run_LLM_inference.py \
            --audio_dir $data_dir/inputs \
            --output_dir $data_dir/outputs/$llm_model \
            --model $llm_model \
            --prompt "$prompt" \
            --split $split \
            --subset $subset \
            --openai-api-key "$openai_api_key" \
            --org "$org" \
            --stop-on-fail &

        sleep 1
    done
done

wait
echo "LLM inference with $llm_model completed for all splits and subsets."

metadata_rows() {
    local metadata_path="$1"
    if [[ ! -f "$metadata_path" ]]; then
        echo "missing"
        return
    fi
    local line_count
    line_count=$(wc -l < "$metadata_path")
    if [[ "$line_count" -gt 0 ]]; then
        echo $((line_count - 1))
    else
        echo 0
    fi
}

echo "Metadata row counts:"
for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        input_metadata="$data_dir/inputs/$split/$subset/metadata.csv"
        output_metadata="$data_dir/outputs/$llm_model/$split/$subset/metadata.csv"
        echo "$split/$subset processed files: $(metadata_rows "$output_metadata")/$(metadata_rows "$input_metadata")"
    done
done

exit
