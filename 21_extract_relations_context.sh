#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh
for split in 'test' 'dev'; do
  for subset in 'improvised' 'naturalistic'; do
    metadata=data/$protocol/inputs/$split/$subset/metadata.csv
    output_dir=data/$protocol/inputs/$split/$subset

    echo "Extract and pickle context and relationship for split:$split subset:$subset"
    $(python_cmd 'SB21' --cpu) bin/naturalness/extract_pickles.py \
        --metadata $metadata \
        --assets-dir $seamless_assets_dir \
        --context-cache-name $output_dir/context_hf_cache.pkl \
        --rel-cache-name $output_dir/relationship_hf_cache.pkl \
        --text-model $sbert_model &

  done
done

wait

echo "all jobs finished!"

exit