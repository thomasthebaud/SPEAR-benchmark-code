#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh
source openai_keys.sh

run_short=0
run_graphs=0
run_long=0
run_summary=0

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
    --summary|--stage4)
      run_summary=1
      ;;
    --all)
      run_short=1
      run_graphs=1
      run_long=1
      run_summary=0
      ;;
    -h|--help)
      cat <<EOF
Usage: bash 61_generate_report.sh [--short] [--graphs] [--long] [--summary] [--all]

Stages:
  --short   Stage 1: write reports/llm_model/report.txt and metrics CSVs
  --graphs  Stage 2: generate matplotlib/seaborn graphs
  --long    Stage 3: generate reports/llm_model/detailed_report.html
  --summary Stage 4: generate reports/llm_model/summary.txt and refresh detailed_report.html
  --all     Run all stages except summary (to avoid using too many tokens unnecessarily)

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

if [[ "$run_short" -eq 0 && "$run_graphs" -eq 0 && "$run_long" -eq 0 && "$run_summary" -eq 0 ]]; then
  run_short=1
  run_graphs=1
  run_long=1
  run_summary=1
fi


if [[ "$run_short" -eq 1 ]]; then
  for model in "${eval_models[@]}"; do
    report_dir="reports/$protocol/$model"
    echo "Stage 1: generating short report for model:$model"
    $(python_cmd 'SB51-S1' --cpu) bin/reports/generate_short_report.py \
        --protocol "$protocol" \
        --model "$model" \
        --asr-models "${asr_models[@]}" \
        --language-id-model "$language_id_model" \
        --dialect-id-model "$dialect_id_model" \
        --stance-model "$stance_llm_model" \
        --sbert-model "${sbert_model:-sentence-transformers/all-MiniLM-L6-v2}" \
        --statistical-test "${statistical_test:-Welch t-test}" \
        --selection-method "end_with_question" \
        --min-turns 2 \
        --min-speakers 2 \
        --data-root "$data_dir" \
        --results-root "results/$protocol" \
        --output-dir "$report_dir" \
        --ignore-features "${ignored_explainable_features[@]:-}" &
  done
  wait
  echo "Short report generation finished for ${eval_models[@]}."

fi

if [[ "$run_graphs" -eq 1 ]]; then
  for model in "${eval_models[@]}"; do
    report_dir="reports/$protocol/$model"
    echo "Stage 2: generating graphs for model:$model"
    $(python_cmd 'SB51-S2' --cpu) bin/reports/generate_graphs.py \
        --model "$model" \
        --results-root "results/$protocol" \
        --output-dir "$report_dir" \
        --ignore-features "${ignored_explainable_features[@]:-}" &
  done

  wait
  echo "Graph generation finished for ${eval_models[@]}."
fi

if [[ "$run_long" -eq 1 ]]; then
  for model in "${eval_models[@]}"; do
    report_dir="reports/$protocol/$model"
    echo "Stage 3: generating detailed HTML report for model:$model"
    python bin/reports/generate_html_report.py \
        --protocol "$protocol" \
        --model "$model" \
        --report-dir "$report_dir" \
        --ignore-features "${ignored_explainable_features[@]:-}" &
  done
  wait
  echo "Detailed HTML report generation finished for ${eval_models[@]}."
fi

if [[ "$run_summary" -eq 1 ]]; then
  if [[ -z "${openai_api_key:-}" ]]; then
    echo "Missing openai_api_key. Set it in openai_keys.sh before running Stage 4." >&2
    exit 1
  fi

  for model in "${eval_models[@]}"; do
    report_dir="reports/$protocol/$model"
    echo "Stage 4: generating summary for model:$model with $summary_llm_model"
    python bin/reports/generate_summary.py \
        --protocol "$protocol" \
        --model "$model" \
        --report-dir "$report_dir" \
        --summary-model "$summary_llm_model" \
        --api-key "$openai_api_key" \
        --organization "${org:-}" &
  done
  wait
  echo "Summary generation finished for ${eval_models[@]}."

  for model in "${eval_models[@]}"; do
    report_dir="reports/$protocol/$model"
    echo "Stage 4: refreshing detailed HTML report with summary for model:$model"
    python bin/reports/generate_html_report.py \
        --protocol "$protocol" \
        --model "$model" \
        --report-dir "$report_dir" \
        --ignore-features "${ignored_explainable_features[@]:-}" &
  done
  wait
  echo "Detailed HTML report refresh finished for ${eval_models[@]}."
fi

echo "report generation finished"

exit
