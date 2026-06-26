#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh

run_intelligibility=0
run_interruptions=0
run_dialects=0
run_emotional_naturalness=0
run_avd=0
run_stances=0
run_explainable=0
run_turntaking_naturalness=0

if [[ $# -eq 0 ]]; then
  run_intelligibility=1
  run_interruptions=1
  run_dialects=1
  run_emotional_naturalness=1
  run_avd=1
  run_stances=1
  run_explainable=1
  run_turntaking_naturalness=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --intelligibility|--stage1)
      run_intelligibility=1
      ;;
    --interruptions|--stage2)
      run_interruptions=1
      ;;
    --dialects|--stage3)
      run_dialects=1
      ;;
    --emotional-naturalness|--stage4)
      run_emotional_naturalness=1
      ;;
    --avd|--stage5)
      run_avd=1
      ;;
    --stances|--stage6)
      run_stances=1
      ;;
    --explainable|--stage7)
      run_explainable=1
      ;;
    --turntaking-naturalness|--wer-length|--stage8)
      run_turntaking_naturalness=1
      ;;
    --all)
      run_intelligibility=1
      run_interruptions=1
      run_dialects=1
      run_emotional_naturalness=1
      run_avd=1
      run_stances=1
      run_explainable=1
      run_turntaking_naturalness=1
      ;;
    -h|--help)
      cat <<EOF
Usage: bash 63_generate_article_graphs.sh [--stage1 ... --stage8] [--all]

Stages:
  --stage1, --intelligibility          Intelligibility and Speech Quality
  --stage2, --interruptions            Interruptions and Latency
  --stage3, --dialects                 Dialects
  --stage4, --emotional-naturalness    Emotional Naturalness
  --stage5, --avd                      AVD consistency
  --stage6, --stances                  Stances
  --stage7, --explainable              EXplainable features
  --stage8, --turntaking-naturalness   Turn-taking naturalness histogram
  --all                                Run all stages

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

graphs_dir="graphs"
results_root="results/$protocol"
mkdir -p "$graphs_dir"

conda_sh="${CONDA_SH:-/home/tthebau1/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "$conda_sh" ]]; then
  echo "Could not find conda setup script: $conda_sh" >&2
  exit 1
fi
source "$conda_sh"
conda activate spearbench

run_graph_stage() {
  local stage_name="$1"
  local script="$2"
  local output="$3"
  local script_num="$4"
  shift 4
  echo "Generating article graph: $stage_name"
  $(python_cmd "SB53-S${script_num}" --cpu) "bin/reports/graphs/$script" \
      --results-root "$results_root" \
      --output-path "$graphs_dir/$output" \
      --models "${eval_models[@]}" \
      --ignore-features "${ignored_explainable_features[@]:-}" \
      "$@" &
}

if [[ "$run_intelligibility" -eq 1 ]]; then
  run_graph_stage "Intelligibility and Speech Quality" "intelligibility_speech_quality.py" "stage1_article_intelligibility_speech_quality.png" "1"
fi

if [[ "$run_interruptions" -eq 1 ]]; then
  run_graph_stage "Interruptions and Latency" "interruptions_latency.py" "stage2_article_interruptions_latency.png" "2"
fi

if [[ "$run_dialects" -eq 1 ]]; then
  run_graph_stage "Dialects" "dialects.py" "stage3_article_dialects.png" "3"
fi

if [[ "$run_emotional_naturalness" -eq 1 ]]; then
  run_graph_stage "Emotional Naturalness" "emotional_naturalness.py" "stage4_article_emotional_naturalness.png" "4"
fi

if [[ "$run_avd" -eq 1 ]]; then
  run_graph_stage "AVD consistency" "avd_consistency.py" "stage5_article_avd_consistency.png" "5"
fi

if [[ "$run_stances" -eq 1 ]]; then
  run_graph_stage "Stances" "stances.py" "stage6_article_stances.png" "6"
fi

if [[ "$run_explainable" -eq 1 ]]; then
  run_graph_stage "EXplainable features" "explainable_features.py" "stage7_article_explainable_features.png" "7"
fi

if [[ "$run_turntaking_naturalness" -eq 1 ]]; then
  run_graph_stage "Turn-taking Naturalness Histogram" "turntaking_naturalness_histogram.py" "stage8_article_turntaking_naturalness_histogram.png" "8"
fi

wait
echo "Article graph generation finished."
