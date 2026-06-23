#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh

missing_only=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --missing-only)
      missing_only=1
      ;;
    -h|--help)
      cat <<EOF
Usage: bash 50_check_missing_files.sh [--missing-only]

Options:
  --missing-only  Only print missing files. By default, print every expected file.
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

splits=(test dev)
subsets=(improvised naturalistic)
models=(original "${eval_models[@]}")
asr_models=(Qwen3-ASR-0.6B whisper-large-v3)
stance_indices=(0) #(0 1 2 3 4 5 6 7 8 9), reduced for readability
stance_indices=(0 1 2 3 4 5 6 7 8 9)
missing_any=0

check_files() {
  local script_id="$1"
  local model="$2"
  local subset_label="$3"
  shift 3

  local missing=()
  local file
  for file in "$@"; do
    if [[ ! -s "$file" ]]; then
      missing+=("$file")
      printf "[ ] script %s\t- model %s\t- subset %s\t- missing: %s\n" "$script_id" "$model" "$subset_label" "$file"
    elif [[ "$missing_only" -eq 0 ]]; then
      printf "[X] script %s\t- model %s\t- subset %s\t- found: %s\n" "$script_id" "$model" "$subset_label" "$file"
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    if [[ "$missing_only" -eq 0 ]]; then
      echo -e "[X] script $script_id\t- model $model\t- subset $subset_label \t- all computed"
    fi
  else
    missing_any=1
  fi
}

# 01_prepare_data_from_seamless.sh: prepare question inputs and original answer metadata.
for split in "${splits[@]}"; do
  for subset in "${subsets[@]}"; do
    check_files "01" "original" "$split/$subset" \
      "data/$protocol/inputs/$split/$subset/metadata.csv" \
      "data/$protocol/outputs/original/$split/$subset/metadata.csv"
  done
done

# 02_run_LLM_inference.sh: generate answer audio and metadata for the configured LLM.
for split in "${splits[@]}"; do
  for subset in "${subsets[@]}"; do
    for model in "${eval_models[@]}"; do
      check_files "02" "$model" "$split/$subset" \
        "data/$protocol/outputs/$model/$split/$subset/metadata.csv"
      done
  done
done

# 03_transcribe.sh: transcribe original and LLM answer audio with each ASR model.
for split in "${splits[@]}"; do
  for subset in "${subsets[@]}"; do
    for model in "${models[@]}"; do
      expected=()
      for asr_model in "${asr_models[@]}"; do
        expected+=("data/$protocol/outputs/$model/$split/$subset/${asr_model}_transcripts.csv")
      done
      check_files "03" "$model" "$split/$subset" "${expected[@]}"
    done
  done
done

# 10_compute_base_metrics.sh: compute WER/CER, latency, interruptions, and UTMOS.
for split in "${splits[@]}"; do
  for subset in "${subsets[@]}"; do
    for model in "${models[@]}"; do
      check_files "10" "$model" "$split/$subset" \
        "results/$protocol/$model/$split/$subset/base_metrics.csv"
    done
  done
done

# 11_language_dialect.sh: predict answer language and dialect and save per-row CSVs.
for split in "${splits[@]}"; do
  for subset in "${subsets[@]}"; do
    for model in "${models[@]}"; do
      check_files "11" "$model" "$split/$subset" \
        "results/$protocol/$model/$split/$subset/language_id.csv" \
        "results/$protocol/$model/$split/$subset/dialect_id.csv"
    done
  done
done

# 20_naturalness_feats.sh: extract question+answer VoxProfile naturalness features and SER_AVD averages.
for split in "${splits[@]}"; do
  for subset in "${subsets[@]}"; do
    for model in "${models[@]}"; do
      check_files "20" "$model" "$split/$subset" \
        "data/$protocol/outputs/$model/$split/$subset/naturalness/voxprofile_features/metadata.csv" \
        "results/$protocol/$model/$split/$subset/SER_AVD.csv"
    done
  done
done

# 21_extract_relations_context.sh: build context and relationship embedding caches.
for split in "${splits[@]}"; do
  for subset in "${subsets[@]}"; do
    check_files "21" "inputs" "$split/$subset" \
      "data/$protocol/inputs/$split/$subset/context_hf_cache.pkl" \
      "data/$protocol/inputs/$split/$subset/relationship_hf_cache.pkl"
  done
done

# 22_score_naturalness.sh: score naturalness using audio features and text caches.
for split in "${splits[@]}"; do
  for subset in "${subsets[@]}"; do
    for model in "${models[@]}"; do
      check_files "22" "$model" "$split/$subset" \
        "results/$protocol/$model/$split/$subset/naturalness_scores.csv" \
        "results/$protocol/$model/$split/$subset/naturalness_scores_normalized.csv" \
        "results/$protocol/$model/$split/$subset/naturalness_predictions_raw.csv"
    done
  done
done

# 30_run_LLM_inference_STANCEs.sh: create STANCE question CSVs and LLM-judge scores.
for split in "${splits[@]}"; do
  subset=improvised
  for model in "${models[@]}"; do
    expected=()
    for idx in "${stance_indices[@]}"; do
      expected+=("data/$protocol/outputs/$model/$split/$subset/stance_questions_Q${idx}.csv")
      expected+=("results/$protocol/$model/$split/$subset/stance_metrics_Q${idx}.csv")
    done
    check_files "30" "$model" "$split/$subset" "${expected[@]}"
  done
done

# 31_compute_STANCE_metrics.sh: merge original and LLM STANCE outputs for comparison.
for split in "${splits[@]}"; do
  subset=improvised
  for model in "${models[@]}"; do
    check_files "31" "$model" "$split/$subset" \
      "results/$protocol/$model/$split/$subset/merged_stances.csv"
  done
done

# 40_extract_explainable_features.sh: extract and normalize explainable distributional baseline features.
for split in "${splits[@]}"; do
  for subset in "${subsets[@]}"; do
    for model in "${models[@]}"; do
      expected=(
        "results/$protocol/$model/$split/$subset/distrib_baselines_features.csv"
        "results/$protocol/$model/$split/$subset/distrib_baselines_features_normalized.csv"
      )
      if [[ "$model" == "original" ]]; then
        expected+=(
          "results/$protocol/$model/$split/$subset/distrib_baselines_features_q.csv"
          "results/$protocol/$model/$split/$subset/distrib_baselines_features_normalized_q.csv"
        )
      fi
      check_files "40" "$model" "$split/$subset" "${expected[@]}"
    done
  done
done

# 41_use_features_for_baseline.sh --scores: train dev-set baselines and score test utterances.
for subset in "${subsets[@]}"; do
  for model in "${models[@]}"; do
    check_files "41-S1" "$model" "test/$subset" \
      "results/$protocol/$model/test/$subset/distrib_baselines_feature_scores.csv" \
      "results/$protocol/$model/distrib_baselines_summary.csv"
  done
done

# 41_use_features_for_baseline.sh --clusters: compute correlation groups and PCA cluster features.
for subset in "${subsets[@]}"; do
  for model in "${models[@]}"; do
    check_files "41-S2" "$model" "test/$subset" \
      "results/$protocol/$model/test/$subset/correlation_feature_groups_rho0p8.csv" \
      "results/$protocol/$model/test/$subset/correlation_cluster_features_rho0p8.csv"
  done
done

# 51_generate_report.sh: generate short reports, per-model graphs, and detailed HTML reports.
report_graphs=(
  basic_metrics.png
  stances.png
  emo_naturalness.png
  emo_naturalness_by_relationship.png
  emotion_scatter.png
  dialect_confusion.png
  dialect_scores.png
  explainables.png
)
for model in "${eval_models[@]}"; do
  expected=(
    "reports/$protocol/$model/report.txt"
    "reports/$protocol/$model/metrics-improvised.csv"
    "reports/$protocol/$model/metrics-naturalistic.csv"
    "reports/$protocol/$model/detailed_report.html"
  )
  for graph in "${report_graphs[@]}"; do
    expected+=("reports/$protocol/$model/graphs/$graph")
  done
  check_files "51" "$model" "reports" "${expected[@]}"
done

# 52_benchmark.sh: generate per-model benchmark CSVs, merged benchmark CSV, and LaTeX table.
expected=(
  "reports/$protocol/benchmark.csv"
  "reports/$protocol/benchmark.tex"
)
for model in "${models[@]}"; do
  expected+=("reports/$protocol/benchmark/$model.csv")
done
check_files "52" "all" "benchmark" "${expected[@]}"

# 53_generate_article_graphs.sh: generate article figures and companion tables.
check_files "53" "all" "graphs" \
  "graphs/stage1_article_intelligibility_speech_quality.png" \
  "graphs/stage2_article_interruptions_latency.png" \
  "graphs/stage2_article_interruptions_latency.tex" \
  "graphs/stage3_article_dialects.png" \
  "graphs/stage3_dialects_nochange.png" \
  "graphs/stage4_article_emotional_naturalness.png" \
  "graphs/stage4_article_emotional_naturalness_short.png" \
  "graphs/stage4_article_emotional_naturalness_violin.png" \
  "graphs/stage5_article_avd_consistency.png" \
  "graphs/stage6_article_stances.png" \
  "graphs/stage6_article_stances_spider.png" \
  "graphs/stage7_article_explainable_features.png" \
  "graphs/stage7_article_explainable_features_combined.png" \
  "graphs/stage8_article_wer_answer_length.png"

exit "$missing_any"
