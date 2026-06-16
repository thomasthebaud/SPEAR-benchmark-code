#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh

run_extract=0
run_normalize=0

if [[ $# -eq 0 ]]; then
  run_extract=1
  run_normalize=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --extract|--stage1)
      run_extract=1
      ;;
    --normalize|--stage2)
      run_normalize=1
      ;;
    --all)
      run_extract=1
      run_normalize=1
      ;;
    -h|--help)
      cat <<EOF
Usage: bash 40_extract_explainable_features.sh [--extract] [--normalize] [--all]

Stages:
  --stage1, --extract      Extract explainable features.
  --stage2, --normalize    Normalize f0 features by answered speaker.
  --all                    Run both stages.

If no stage is passed, both stages are run.
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

if [[ "$run_extract" -eq 1 ]]; then
  # for split in 'test' 'dev'; do
  #   for subset in 'improvised' 'naturalistic'; do
  #     for model in "original" "${eval_models[@]}"; do
  #       metadata=data/$protocol/outputs/$model/$split/$subset/metadata.csv
  #       output=results/$protocol/$model/$split/$subset/distrib_baselines_features.csv

  #       echo "Extract explainable features for split:$split subset:$subset model:$model"
  #       $(python_cmd 'SB40-S1' --cpu) bin/distrib_baselines/extract_features.py \
  #           --metadata "$metadata" \
  #           --relationships-csv "$seamless_assets_dir/relationships.csv" \
  #           --output "$output" &

  #     done
  #   done
  # done

  for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
      model="original"
      metadata=data/$protocol/outputs/$model/$split/$subset/metadata.csv
      output=results/$protocol/$model/$split/$subset/distrib_baselines_features_q.csv

      echo "Extract explainable features for the questions of split:$split subset:$subset"
      $(python_cmd 'SB40-S1' --cpu) bin/distrib_baselines/extract_features.py \
          --metadata "$metadata" \
          --relationships-csv "$seamless_assets_dir/relationships.csv" \
          --output "$output" \
          --questions &

    done
  done

  wait
fi

if [[ "$run_normalize" -eq 1 ]]; then
  # for split in 'test' 'dev'; do
  #   for subset in 'improvised' 'naturalistic'; do
  #     for model in "${eval_models[@]}"; do
  #       metadata=data/$protocol/outputs/$model/$split/$subset/metadata.csv
  #       features=results/$protocol/$model/$split/$subset/distrib_baselines_features.csv
  #       output=results/$protocol/$model/$split/$subset/distrib_baselines_features_normalized.csv

  #       echo "Normalize explainable features for split:$split subset:$subset model:$model"
  #       $(python_cmd 'SB40-S2' --cpu) bin/distrib_baselines/normalize_features_by_answered_speaker.py \
  #           --metadata "$metadata" \
  #           --features "$features" \
  #           --output "$output" &

  #     done
  #   done
  # done

  for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
      model="original"
      metadata=data/$protocol/outputs/$model/$split/$subset/metadata.csv
      features=results/$protocol/$model/$split/$subset/distrib_baselines_features_q.csv
      output=results/$protocol/$model/$split/$subset/distrib_baselines_features_normalized_q.csv

      echo "Normalize explainable features of the questions for split:$split subset:$subset"
      $(python_cmd 'SB40-S2' --cpu) bin/distrib_baselines/normalize_features_by_answered_speaker.py \
          --metadata "$metadata" \
          --features "$features" \
          --output "$output" \
          --questions &

    done
  done

  wait
fi


echo "all jobs finished!"

exit
