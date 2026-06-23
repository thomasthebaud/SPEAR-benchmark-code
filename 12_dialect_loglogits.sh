#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh

echo "Projecting dialect log-logits for protocol $protocol"

mkdir -p "results/$protocol/dialect_logits" graphs

dialect_logit_models=()
for model in original "${eval_models[@]}"; do
  already_added=0
  for existing in "${dialect_logit_models[@]}"; do
    if [[ "$existing" == "$model" ]]; then
      already_added=1
      break
    fi
  done
  if [[ "$already_added" -eq 0 ]]; then
    dialect_logit_models+=("$model")
  fi
done

echo "Models: ${dialect_logit_models[*]}"
graph_root='graphs/dialect_logits'

$(python_cmd 'SB12' --cpu) bin/voxlect/dialect_loglogits.py \
  --results-dir "results/$protocol" \
  --models "${dialect_logit_models[@]}" \
  --output-dir "results/$protocol/dialect_logits" \
  --graph-root "$graph_root"

exit
