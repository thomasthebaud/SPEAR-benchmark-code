#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source openai_keys.sh
source cmd.sh

api_key="$openai_api_key"
if [[ "$llm_model" == gemini-* ]]; then
    api_key="$gemini_api_key"
fi

num_gpus="${N_GPUS:-1}"
execution_flag="${EXECUTION_FLAG:---gpu}"

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
Usage: bash 02b_run_parallel_LLM_inference.sh [--num-gpus N] [--gpu|--gpu-a100]

For each split/subset, launches N one-GPU jobs. Each job receives 1/N of the
input metadata and writes a shard metadata CSV; shards are merged into the
normal metadata.csv after the N jobs finish. You can also set N_GPUS=N.
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

prompt="You are participating in a natural spoken conversation.\
    Answer when it feels natural, not only at the very end.\
    Keep responses conversational and concise.\
    If the user interrupts, stop and respond to the latest user speech."

if [[ "$llm_model" == "mini-omni" ]]; then
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
    export MINI_OMNI_SYNC_CUDA="${MINI_OMNI_SYNC_CUDA:-1}"
    export MINI_OMNI_RAISE_ERRORS="${MINI_OMNI_RAISE_ERRORS:-0}"
fi

echo "Running sharded parallel inference with model $llm_model on seamless interactions"
echo "Input data directory: $data_dir/inputs"
echo "Output data directory: $data_dir/outputs/$llm_model"
echo "GPU shards per split/subset: $num_gpus ($execution_flag)"

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

run_one_shard() {
    local split="$1"
    local subset="$2"
    local shard_index="$3"
    local output_csv_name="metadata_shard_${shard_index}_of_${num_gpus}.csv"
    echo "on split $split subset $subset shard $((shard_index + 1))/$num_gpus"
    $(python_cmd "SB02-${split:0:1}${subset:0:1}-${shard_index}" "$execution_flag") bin/run_LLM_inference.py \
        --audio_dir "$data_dir/inputs" \
        --output_dir "$data_dir/outputs/$llm_model" \
        --model "$llm_model" \
        --prompt "$prompt" \
        --split "$split" \
        --subset "$subset" \
        --openai-api-key "$api_key" \
        --org "$org" \
        --shard-index "$shard_index" \
        --num-shards "$num_gpus" \
        --output-csv-name "$output_csv_name"
}

merge_shards() {
    local split="$1"
    local subset="$2"
    local subset_output_dir="$data_dir/outputs/$llm_model/$split/$subset"
    local output_metadata="$subset_output_dir/metadata.csv"
    python3 - "$subset_output_dir" "$num_gpus" "$output_metadata" <<'PYMERGE'
import sys
from pathlib import Path
import pandas as pd

subset_output_dir = Path(sys.argv[1])
num_shards = int(sys.argv[2])
output_metadata = Path(sys.argv[3])
frames = []
if output_metadata.exists():
    existing = pd.read_csv(output_metadata)
    existing = existing.loc[:, ~existing.columns.str.startswith("Unnamed:")]
    frames.append(existing)
for shard_index in range(num_shards):
    shard_path = subset_output_dir / f"metadata_shard_{shard_index}_of_{num_shards}.csv"
    if not shard_path.exists():
        print(f"[WARN] Missing shard metadata: {shard_path}", flush=True)
        continue
    frame = pd.read_csv(shard_path)
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed:")]
    frames.append(frame)
if frames:
    merged = pd.concat(frames, ignore_index=True)
    if "audio_path" in merged.columns:
        merged = merged.drop_duplicates(subset=["audio_path"], keep="last")
        merged = merged.sort_values("audio_path").reset_index(drop=True)
else:
    merged = pd.DataFrame()
output_metadata.parent.mkdir(parents=True, exist_ok=True)
tmp_path = output_metadata.with_suffix(output_metadata.suffix + ".tmp")
merged.to_csv(tmp_path, index=False)
tmp_path.replace(output_metadata)
print(f"Merged {len(merged)} rows into {output_metadata}", flush=True)
PYMERGE
}

for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        echo "Launching $num_gpus shards for $split/$subset"
        for shard_index in $(seq 0 $((num_gpus - 1))); do
            run_one_shard "$split" "$subset" "$shard_index" &
        done
        wait
        merge_shards "$split" "$subset"
    done
done

echo "Sharded parallel LLM inference with $llm_model completed for all splits and subsets."

echo "Metadata row counts:"
for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        input_metadata="$data_dir/inputs/$split/$subset/metadata.csv"
        output_metadata="$data_dir/outputs/$llm_model/$split/$subset/metadata.csv"
        echo "$split/$subset processed files: $(metadata_rows "$output_metadata")/$(metadata_rows "$input_metadata")"
    done
done

exit
