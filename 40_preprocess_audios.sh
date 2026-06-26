#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh

num_shards="1"
shard_index="0"
limit="0"
overwrite="0"

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
echo "WhisperX model=$whisperx_model"

# #To kill : squeue -h -u "$USER" -o "%i %j" | awk '$2 ~ /^SB40/ {print $1}' | xargs -r scancel


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
        --language "en" \
        --device "cuda" \
        --compute-type "int8" \
        --align-device "cpu" \
        --batch-size "16" \
        --shard-index "$shard_index" \
        --num-shards "$num_shards" \
        --failures-jsonl "$output_dir/transcription_failures.jsonl" \
        "${extra_args[@]}" &
    done
  done
done

# for split in 'test' 'dev'; do
#     for subset in 'improvised' 'naturalistic'; do
#       model="original"
#       metadata="data/$protocol/outputs/$model/$split/$subset/metadata.csv"
#       output_dir="data/$protocol/outputs/$model/$split/$subset/baseline_prepreprocess"

#       echo "Annotating questions of split:$split subset:$subset"
#       $(python_cmd 'SB40-Q' --gpu) bin/distrib_baselines/preprocess_audio_alignments.py \
#         --metadata "$metadata" \
#         --output-dir "$output_dir" \
#         --audio-column audio_path \
#         --model "$whisperx_model" \
#         --language "en" \
#         --device "cuda" \
#         --compute-type "int8" \
#         --align-device "cpu" \
#         --batch-size "16" \
#         --shard-index "$shard_index" \
#         --num-shards "$num_shards" \
#         --failures-jsonl "$output_dir/transcription_failures.jsonl" \
#         "${extra_args[@]}" &

#     done
#   done

wait
echo "Audio preprocessing finished."
