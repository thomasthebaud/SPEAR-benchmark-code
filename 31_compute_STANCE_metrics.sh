#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source openai_keys.sh

for split in 'test'; do
    for subset in 'improvised'; do
        metrics="results/$protocol/$llm_model/$split/$subset"
            
        srun -p cpu \
            python3 bin/STANCE/merge_outputs.py \
            --stances-original "results/$protocol/original/$split/$subset" \
            --stances-llm "results/$protocol/$llm_model/$split/$subset" \
            --output-csv "$metrics/merged_stances.csv"
            

    done
done

  
