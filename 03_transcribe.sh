#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh

num_gpus="10"
execution_flag="--gpu"
complete_exit_code=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus|-n)
            num_gpus="$2"
            shift 2
            ;;
        --gpu|--gpu-a100)
            execution_flag="$1"
            shift
            ;;
        -h|--help)
            cat <<EOF
Usage: bash 03_transcribe.sh [--num-gpus N] [--gpu|--gpu-a100]

Launches N one-GPU transcription shards for each model/split/subset/ASR
combination, then merges shard transcript CSVs into the normal ASR transcript
CSV. You can also set N_GPUS=N.
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if ! [[ "$num_gpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-gpus must be a positive integer, got: $num_gpus" >&2
    exit 2
fi

echo "Transcribing outputs of model $llm_model"
echo "data directory: $data_dir"
echo "GPU shards per model/split/subset/ASR: $num_gpus ($execution_flag)"

needs_transcription() {
    local model="$1"
    local split="$2"
    local subset="$3"
    local asr="$4"
    local asr_name="${asr%/}"
    asr_name="${asr_name##*/}"
    asr_name="${asr_name// /_}"

    $(python_cmd "SB03-check-${asr_name}" --cpu) bin/check_transcription_needed.py \
        --input-metadata "data/$protocol/outputs/$model/$split/$subset/metadata.csv" \
        --output-path "data/$protocol/outputs/$model/$split/$subset/${asr_name}_transcripts.csv"
}

run_one_shard() {
    local model="$1"
    local split="$2"
    local subset="$3"
    local asr="$4"
    local shard_index="$5"
    local asr_name="${asr%/}"
    asr_name="${asr_name##*/}"
    asr_name="${asr_name// /_}"
    local output_csv_name="${asr_name}_transcripts_shard_${shard_index}_of_${num_gpus}.csv"

    echo "running transcription for $model $split $subset ASR model=$asr shard $((shard_index + 1))/$num_gpus"
    $(python_cmd "SB03-${asr_name}-${shard_index}" "$execution_flag") bin/transcribe.py \
        --data_dir "data/$protocol/outputs/$model" \
        --model "$asr" \
        --split "$split" \
        --subset "$subset" \
        --output_dir "data/$protocol/outputs/$model/$split/$subset" \
        --shard-index "$shard_index" \
        --num-shards "$num_gpus" \
        --output-csv-name "$output_csv_name"
}

merge_shards() {
    local model="$1"
    local split="$2"
    local subset="$3"
    local asr="$4"

    python3 bin/merge_transcript_shards.py \
        --output-dir "data/$protocol/outputs/$model/$split/$subset" \
        --asr "$asr" \
        --num-shards "$num_gpus"
}

for split in "test" "dev"; do
    for subset in "improvised" "naturalistic"; do
        for model in "${eval_models[@]}"; do
            for asr in "${asr_models[@]}"; do
                if needs_transcription "$model" "$split" "$subset" "$asr"; then
                    :
                else
                    check_status=$?
                    if [[ "$check_status" -eq "$complete_exit_code" ]]; then
                        continue
                    fi
                    exit "$check_status"
                fi

                echo "Launching $num_gpus transcription shards for $model $split/$subset ASR model=$asr"
                for shard_index in $(seq 0 $((num_gpus - 1))); do
                    run_one_shard "$model" "$split" "$subset" "$asr" "$shard_index" &
                done
                wait
                merge_shards "$model" "$split" "$subset" "$asr"
            done
        done
    done
done

echo "All sharded transcriptions completed for original and $llm_model outputs with ${asr_models[@]} ASR models."
exit
