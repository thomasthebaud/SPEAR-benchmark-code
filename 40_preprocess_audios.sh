#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh

whisperx_model="${WHISPERX_MODEL:-large-v2}"
whisperx_language="${WHISPERX_LANGUAGE:-en}"
whisperx_device="${WHISPERX_DEVICE:-cuda}"
whisperx_compute_type="${WHISPERX_COMPUTE_TYPE:-int8}"
whisperx_align_device="${WHISPERX_ALIGN_DEVICE:-cpu}"
whisperx_batch_size="${WHISPERX_BATCH_SIZE:-16}"
num_shards="${NUM_SHARDS:-1}"
shard_index="${SHARD_INDEX:-0}"
limit="${LIMIT:-0}"
overwrite="${OVERWRITE:-0}"

extra_args=()
if [[ "$overwrite" == "1" ]]; then
  extra_args+=(--overwrite)
fi
if [[ "$limit" != "0" ]]; then
  extra_args+=(--limit "$limit")
fi

models=("original" "${eval_models[@]}")
models=("${eval_models[@]}")
echo "Preprocessing answer audio for distrib baselines"
echo "protocol=$protocol"
echo "models=${models[*]}"
echo "WhisperX model=$whisperx_model device=$whisperx_device compute_type=$whisperx_compute_type align_device=$whisperx_align_device"

for split in 'test' 'dev'; do
  for subset in 'improvised' 'naturalistic'; do
    for model in "${models[@]}"; do
      metadata="data/$protocol/outputs/$model/$split/$subset/metadata.csv"
      output_dir="data/$protocol/outputs/$model/$split/$subset/baseline_prepreprocess"

      if [[ ! -f "$metadata" ]]; then
        echo "Skipping missing metadata: $metadata"
        continue
      fi

      echo "Annotating answers for model=$model split=$split subset=$subset"
      $(python_cmd 'SB40-A' --gpu) bin/distrib_baselines/preprocess_audio_alignments.py \
        --metadata "$metadata" \
        --output-dir "$output_dir" \
        --audio-column answer_audio_path \
        --model "$whisperx_model" \
        --language "$whisperx_language" \
        --device "$whisperx_device" \
        --compute-type "$whisperx_compute_type" \
        --align-device "$whisperx_align_device" \
        --batch-size "$whisperx_batch_size" \
        --shard-index "$shard_index" \
        --num-shards "$num_shards" \
        --failures-jsonl "$output_dir/transcription_failures.jsonl" \
        "${extra_args[@]}" &
    done
  done
done

for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
      model="original"
      metadata="data/$protocol/outputs/$model/$split/$subset/metadata.csv"
      output_dir="data/$protocol/outputs/$model/$split/$subset/baseline_prepreprocess"

      echo "Extract explainable features for the questions of split:$split subset:$subset"
      $(python_cmd 'SB40-Q' --gpu) bin/distrib_baselines/preprocess_audio_alignments.py \
        --metadata "$metadata" \
        --output-dir "$output_dir" \
        --audio-column audio_path \
        --model "$whisperx_model" \
        --language "$whisperx_language" \
        --device "$whisperx_device" \
        --compute-type "$whisperx_compute_type" \
        --align-device "$whisperx_align_device" \
        --batch-size "$whisperx_batch_size" \
        --shard-index "$shard_index" \
        --num-shards "$num_shards" \
        --failures-jsonl "$output_dir/transcription_failures.jsonl" \
        "${extra_args[@]}" &

    done
  done

wait
echo "Audio preprocessing finished."
