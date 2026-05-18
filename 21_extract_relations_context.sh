#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
for split in 'test' 'dev'; do
  for subset in 'improvised' 'naturalistic'; do
    metadata=data/$protocol/inputs/$split/$subset/${split}_${subset}_metadata.csv
    output_dir=data/$protocol/inputs/$split/$subset

    echo "Extract and pickle context and relationship for split:$split subset:$subset"
    srun -p cpu \
      python3 bin/naturalness/extract_pickles.py \
        --metadata $metadata \
        --assets-dir $seamless_assets_dir \
        --context-cache-name $output_dir/context_hf_cache.pkl \
        --rel-cache-name $output_dir/relationship_hf_cache.pkl &

  done
done

wait

echo "all jobs finished!"

exit