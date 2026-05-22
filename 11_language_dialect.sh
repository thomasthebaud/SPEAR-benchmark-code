#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh

run_LID=0
run_DialectID=0
run_Summary=0

if [[ $# -eq 0 ]]; then
  run_LID=1
  run_DialectID=1
  run_Summary=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --language|--stage1)
      run_LID=1
      ;;
    --dialect|--stage2)
      run_DialectID=1
      ;;
    --summary|--stage3)
      run_Summary=1
      ;;
    --all)
      run_LID=1
      run_DialectID=1
      run_Summary=1
      ;;
    -h|--help)
      cat <<EOF
Usage: bash 11_language_dialect.sh [--language] [--dialect] [--summary] [--all]

Stages:
  --language   Stage 1: predict output languages
  --dialect    Stage 2: predict dialects
  --summary    Stage 3: summarize language and dialect outputs

If no stage is passed, all stages are run.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done


if [[ "$run_LID" -eq 1 ]];then
    echo "Stage 1:Predicting output languages of model $llm_model using $language_id_model"

    # Stage 1: predict the language of each model output.
    for split in 'test' 'dev'; do
        for subset in 'improvised' 'naturalistic'; do
            for model in 'original' $llm_model; do
                echo "### Running language ID for $model $split $subset ###"
                metadata=data/$protocol/outputs/$model/$split/$subset/${split}_${subset}_metadata.csv
                output_dir=results/$protocol/$model/$split/$subset
                output=$output_dir/language_id.csv

                mkdir -p "$output_dir"
                srun -p gpu --gpus 1 python3 bin/language_id.py \
                    --metadatafile "$metadata" \
                    --output "$output" \
                    --model "$language_id_model" \
                    --force-recompute &
            done
        done
    done

    wait

fi

if [[ "$run_DialectID" -eq 1 ]];then
    echo "Stage 2:Predicting output dialects of model $llm_model using voxlect models"

    # Stage 2: predict the dialect of each model output.
    for split in 'test' 'dev'; do
        for subset in 'improvised' 'naturalistic'; do
            for model in 'original' $llm_model; do
                echo "### Running dialect ID for $model $split $subset ###"
                metadata=data/$protocol/outputs/$model/$split/$subset/${split}_${subset}_metadata.csv
                output_dir=results/$protocol/$model/$split/$subset
                output=$output_dir/dialect_id.csv

                mkdir -p "$output_dir"
                srun -p gpu --gpus 1 python3 bin/dialect_id.py \
                    --metadatafile "$metadata" \
                    --output "$output" \
                    --language "english" \
                    --languagefile $output_dir/language_id.csv \
                    --force-recompute &

            done
        done
    done

    wait

fi

if [[ "$run_Summary" -eq 1 ]];then
    echo "Stage 3: Summarizing language and dialect outputs" >&2

    # Stage 3: print a compact summary for every split/subset/model result.
    python3 bin/language_dialect_summary.py \
        --results-dir "results/$protocol" \
        --models original "$llm_model"
fi

exit
