#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source openai_keys.sh
source cmd.sh

for split in 'test' 'dev'; do
    for subset in 'improvised'; do
        for model in 'original' $llm_model; do
            echo "### Merging STANCE metrics for $model $split $subset ###"
            metrics="results/$protocol/$model/$split/$subset"
                
            $(python_cmd 'SB31' --cpu) bin/STANCE/merge_outputs.py \
                --stances-original "results/$protocol/original/$split/$subset" \
                --stances-llm "results/$protocol/$model/$split/$subset" \
                --output-csv "$metrics/merged_stances.csv" &
        
        done    
    done
done

wait
echo "All STANCE metrics computed for original and $llm_model outputs."
exit
  
