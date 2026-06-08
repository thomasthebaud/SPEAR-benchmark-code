#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh

echo "Verifying file integrity for protocol $protocol"
echo "data directory: $data_dir"

for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        inputs_metadata="$data_dir/inputs/${split}/${subset}/metadata.csv"
        for model in "${eval_models[@]}"; do
            metadata="$data_dir/outputs/$model/$split/$subset/metadata.csv"
            output="$data_dir/outputs/$model/$split/$subset/metadata_verified.csv"
            if [[ ! -f "$metadata" ]]; then
                echo "[WARN] Missing metadata for $model $split $subset: $metadata"
                continue
            fi
            if [[ ! -f "$inputs_metadata" ]]; then
                echo "[WARN] Missing inputs metadata for $split $subset: $inputs_metadata"
                continue
            fi
            echo "Verifying $model $split $subset"
            $(python_cmd 'SB04' --cpu) bin/verify_file_integrity.py \
                        --metadata "$metadata" \
                        --inputs-metadata "$inputs_metadata" \
                        --base-dir "$(pwd)" \
                        --output "$output" &
        done
    done
done

wait
echo "File integrity verification finished."

# echo "Promoting verified metadata files when safe."
# for split in 'test' 'dev'; do
#     for subset in 'improvised' 'naturalistic'; do
#         for model in "${eval_models[@]}"; do
#             metadata_dir="$data_dir/outputs/$model/$split/$subset"
#             metadata="$metadata_dir/metadata.csv"
#             metadata_old="$metadata_dir/metadata.old.csv"
#             metadata_verified="$metadata_dir/metadata_verified.csv"
#             if [[ -f "$metadata_verified" ]]; then
#                 if [[ -f "$metadata_old" ]]; then
#                     echo "[WARN] Not promoting $metadata_verified because $metadata_old already exists"
#                     continue
#                 fi
#                 if [[ ! -f "$metadata" ]]; then
#                     echo "[WARN] Not promoting $metadata_verified because $metadata is missing"
#                     continue
#                 fi
#                 echo "Promoting verified metadata for $model $split $subset"
#                 mv "$metadata" "$metadata_old"
#                 mv "$metadata_verified" "$metadata"
#             fi
#         done
#     done
# done

# echo "Metadata promotion finished."
