#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh

run_extract=false
run_aggregate=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --extract|--stage1)
            run_extract=true
            shift
            ;;
        --aggregate|--stage2)
            run_aggregate=true
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--extract] [--aggregate] [--force-recompute]" >&2
            exit 2
            ;;
    esac
done

if [[ $run_extract == false && $run_aggregate == false ]]; then
    run_extract=true
    run_aggregate=true
fi

if [[ $run_extract == true ]]; then
    for split in 'test' 'dev'; do
        for subset in 'improvised' 'naturalistic'; do
            for model in 'original' $llm_model; do
              metadata=data/$protocol/outputs/$model/$split/$subset/metadata.csv

              echo "Extract features for split $split subset $subset model $model"

              $(python_cmd 'SB20-S1' --gpu) bin/naturalness/extract_features.py \
                  --metadata $metadata \
                  --min-len-question 3.0 \
                  --force-recompute &

                sleep 1

            done
        done
    done

    wait
    echo "All naturalness features extracted for original and $llm_model outputs."
fi

if [[ $run_aggregate == true ]]; then
    for split in 'test' 'dev'; do
        for subset in 'improvised' 'naturalistic'; do
            for model in 'original' $llm_model; do
              metadata=data/$protocol/outputs/$model/$split/$subset/metadata.csv
              metrics=results/$protocol/$model/$split/$subset

              echo "Export full-turn AVD emotions for split $split subset $subset model $model"

              $(python_cmd 'SB20-S2' --cpu) bin/naturalness/aggregate_AVD_emotions.py \
                  --metadata "$metadata" \
                  --output "$metrics/SER_AVD.csv" &

            done
        done
    done

    wait
    echo "All full-turn SER AVD scores saved for original and $llm_model outputs."
fi
exit
