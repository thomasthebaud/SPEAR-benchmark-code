#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh

run_short=1
run_graphs=1
run_long=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --short|--stage1)
      run_short=1
      ;;
    --graphs|--stage2)
      run_graphs=1
      ;;
    --long|--stage3)
      run_long=1
      ;;
    --all)
      run_short=1
      run_graphs=1
      run_long=1
      ;;
    -h|--help)
      cat <<EOF
Usage: bash 50_generate_report.sh [--short] [--graphs] [--long] [--all]

Stages:
  --short   Stage 1: write reports/\$llm_model/report.txt and metrics.csv
  --graphs  Stage 2: generate matplotlib/seaborn graphs
  --long    Stage 3: generate reports/$llm_model/detailed_report.html

If no stage is passed, --short is run.
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

report_dir="reports/$llm_model"

if [[ "$run_short" -eq 1 ]]; then
  echo "Stage 1: generating short report for model:$llm_model"
  $(python_cmd 'SB51-S1' --cpu) bin/reports/generate_short_report.py \
      --protocol "$protocol" \
      --model "$llm_model" \
      --asr-model "$asr" \
      --stance-model "$stance_llm_model" \
      --sbert-model "${sbert_model:-sentence-transformers/all-MiniLM-L6-v2}" \
      --statistical-test "${statistical_test:-Welch t-test}" \
      --selection-method "end_with_question" \
      --min-turns 1 \
      --min-speakers 1 \
      --data-root "$data_dir" \
      --results-root "results/$protocol" \
      --output-dir "$report_dir"
fi

if [[ "$run_graphs" -eq 1 ]]; then
  echo "Stage 2: generating graphs for model:$llm_model"
  $(python_cmd 'SB51-S2' --cpu) bin/reports/generate_graphs.py       --model "$llm_model"       --results-root "results/$protocol"       --output-dir "$report_dir"
fi

if [[ "$run_long" -eq 1 ]]; then
  echo "Stage 3: generating detailed HTML report for model:$llm_model"
  $(python_cmd 'SB51-S3' --cpu) bin/reports/generate_html_report.py       --protocol "$protocol"       --model "$llm_model"       --report-dir "$report_dir"
fi

echo "report generation finished"

exit
