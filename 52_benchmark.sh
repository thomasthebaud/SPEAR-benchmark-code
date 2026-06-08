#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh

run_lines=0
run_merge=0
n_digits=3

if [[ $# -eq 0 ]]; then
  run_lines=1
  run_merge=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lines|--stage1)
      run_lines=1
      ;;
    --merge|--stage2)
      run_merge=1
      ;;
    --all)
      run_lines=1
      run_merge=1
      ;;
    --n-digits)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --n-digits" >&2
        exit 2
      fi
      n_digits="$2"
      shift
      ;;
    -h|--help)
      cat <<EOF
Usage: bash 52_benchmark.sh [--lines] [--merge] [--all] [--n-digits N]

Stages:
  --lines    Stage 1: generate one benchmark CSV line per model
  --merge    Stage 2: merge per-model benchmark CSVs into reports/$protocol/benchmark.csv
  --all      Run both stages
  --n-digits Digits to keep after the decimal point in stage-1 CSVs (default: 3)

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

benchmark_dir="reports/$protocol/benchmark"
benchmark_csv="reports/$protocol/benchmark.csv"

if [[ "$run_lines" -eq 1 ]]; then
  benchmark_models=("original" "${eval_models[@]}")
  for llm_model in "${benchmark_models[@]}";
  do
      echo "Generating the benchmark line for model $llm_model"
      $(python_cmd 'SB52' --cpu) bin/reports/benchmark_line.py \
          --protocol "$protocol" \
          --model "$llm_model" \
          --results-root "results/$protocol" \
          --output-csv "$benchmark_dir/$llm_model.csv" \
          --n-digits "$n_digits" \
          --ignore-features "${ignored_explainable_features[@]:-}" &

  done

  wait
  echo "All benchmark lines generated for models: ${benchmark_models[@]}"
fi

if [[ "$run_merge" -eq 1 ]]; then
  echo "Merging benchmark lines from $benchmark_dir into $benchmark_csv"
  mkdir -p "$(dirname "$benchmark_csv")"
  shopt -s nullglob
  benchmark_lines=("$benchmark_dir"/*.csv)
  shopt -u nullglob

  if [[ "${#benchmark_lines[@]}" -eq 0 ]]; then
    echo "No benchmark CSVs found in $benchmark_dir" >&2
    exit 1
  fi

  first=1
  : > "$benchmark_csv"
  for csv in "${benchmark_lines[@]}"; do
    if [[ "$first" -eq 1 ]]; then
      cat "$csv" > "$benchmark_csv"
      first=0
    else
      tail -n +2 "$csv" >> "$benchmark_csv"
    fi
  done
  echo "Wrote merged benchmark CSV: $benchmark_csv"
fi
