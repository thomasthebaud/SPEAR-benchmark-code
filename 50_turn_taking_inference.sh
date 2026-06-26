#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh

experiment="group4-dualturn-full-all6-fvad256"
models_to_score=("original" "${eval_models[@]}")
models_to_score=("${eval_models[@]}")
for split in 'test' 'dev'; do
  for subset in 'improvised' 'naturalistic'; do
    for model in "${models_to_score[@]}"; do
      metadata="data/$protocol/outputs/$model/$split/$subset/metadata.csv"
      output_dir="results/$protocol/$model/$split/$subset"
      output_csv="$output_dir/turntaking.${experiment}.csv"

      if [[ ! -f "$metadata" ]]; then
        echo "Warning: missing metadata for split:$split subset:$subset model:$model at $metadata; skipping" >&2
        continue
      fi

      mkdir -p "$output_dir"
      echo "Turn-taking inference for split:$split subset:$subset model:$model"
      $(python_cmd "SB50-${model}" --gpu) bin/turntaking/run_turntaking_inference.py \
          --metadata "$metadata" \
          --output "$output_csv" \
          --checkpoint "models/turntaking_checkpoints/${experiment}.pt" \
          --experiment "$experiment" \
          --batch-size 128 \
          --vad-source "silero" \
          --base-dir "$(pwd)" &

    done
  done
done

wait
echo "All turn-taking inference jobs finished."
exit
