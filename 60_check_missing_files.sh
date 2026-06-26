#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh

missing_only=0
light=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --missing-only)
      missing_only=1
      ;;
    --light)
      light=1
      ;;
    -h|--help)
      cat <<EOF
Usage: bash 60_check_missing_files.sh [--missing-only] [--light]

Options:
  --missing-only  Alias for the default behavior: only print missing or incomplete files.
  --light         Only print missing/empty files; skip CSV row-count completeness checks.
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
models=("${eval_models[@]}")

asr_models=(Qwen3-ASR-0.6B whisper-large-v3)
stance_indices=(0) #(0 1 2 3 4 5 6 7 8 9), reduced for readability
stance_indices=(0 1 2 3 4 5 6 7 8 9)
missing_any=0

csv_data_rows() {
  local file="$1"
  if [[ ! -s "$file" ]]; then
    echo 0
    return
  fi
  local lines
  lines=$(wc -l < "$file" | tr -d ' ')
  if [[ -z "$lines" || "$lines" -le 0 ]]; then
    echo 0
  else
    echo $((lines - 1))
  fi
}

naturalness_completed_pairs() {
  local file="$1"
  python - "$file" <<'PYCOUNT'
import csv
import sys
from collections import defaultdict

path = sys.argv[1]
required = {"question", "answer"}
sources_by_pair = defaultdict(set)
try:
    with open(path, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if not reader.fieldnames or "pair_stem" not in reader.fieldnames:
            print(0)
            raise SystemExit
        for row in reader:
            key = str(row.get("pair_stem", "")).strip().lower()
            if not key:
                continue
            source = str(row.get("vad_source", "")).strip().lower()
            if source.startswith("existing_"):
                source = source.removeprefix("existing_")
            if source in required:
                sources_by_pair[key].add(source)
except Exception:
    print(0)
    raise SystemExit
print(sum(1 for sources in sources_by_pair.values() if sources == required))
PYCOUNT
}

reference_metadata_for() {
  local model="$1"
  local subset_label="$2"
  if [[ "$subset_label" != */* ]]; then
    return 1
  fi
  if [[ "$model" == "all" || "$model" == "inputs" ]]; then
    return 1
  fi
  local split="${subset_label%%/*}"
  local subset="${subset_label#*/}"
  local ref="data/$protocol/outputs/$model/$split/$subset/metadata.csv"
  if [[ -s "$ref" ]]; then
    printf "%s" "$ref"
    return 0
  fi
  return 1
}

check_csv_rows() {
  local script_id="$1"
  local model="$2"
  local subset_label="$3"
  local file="$4"

  [[ "$light" -eq 0 ]] || return 0
  [[ "$file" == *.csv ]] || return 0
  [[ -s "$file" ]] || return 0

  local ref
  if [[ "$script_id" == "30" && "$file" == */stance_metrics_Q*.csv ]]; then
    ref="${file/results\/$protocol/data\/$protocol\/outputs}"
    ref="${ref/stance_metrics_/stance_questions_}"
  elif ! ref=$(reference_metadata_for "$model" "$subset_label"); then
    return 0
  fi
  if [[ ! -s "$ref" ]]; then
    return 0
  fi

  local ref_rows
  ref_rows=$(csv_data_rows "$ref")

  if [[ "$file" == *"/naturalness/voxprofile_features/metadata.csv" ]]; then
    local file_rows completed_pairs missing_pairs
    file_rows=$(csv_data_rows "$file")
    completed_pairs=$(naturalness_completed_pairs "$file")
    missing_pairs=$((ref_rows - completed_pairs))
    if [[ "$missing_pairs" -lt 0 ]]; then
      missing_pairs=0
    fi
    if [[ "$missing_pairs" -gt 0 ]]; then
      printf "[#] script %s\t- model %s\t- subset %s\t- incomplete: rows: %s; completed pairs: %s/%s; missing pairs: %s; file: %s\n" \
        "$script_id" "$model" "$subset_label" "$file_rows" "$completed_pairs" "$ref_rows" "$missing_pairs" "$file"
    fi
    if [[ "$missing_pairs" -gt 0 ]]; then
      missing_any=1
    fi
    return 0
  fi

  local file_rows missing_rows
  file_rows=$(csv_data_rows "$file")
  missing_rows=$((ref_rows - file_rows))
  if [[ "$missing_rows" -lt 0 ]]; then
    missing_rows=0
  fi
  if [[ "$missing_rows" -gt 0 ]]; then
    printf "[#] script %s\t- model %s\t- subset %s\t- incomplete: rows: %s/%s; missing rows: %s; file: %s\n" \
      "$script_id" "$model" "$subset_label" "$file_rows" "$ref_rows" "$missing_rows" "$file"
  fi
  if [[ "$missing_rows" -gt 0 ]]; then
    missing_any=1
  fi
}

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
    fi
    check_csv_rows "$script_id" "$model" "$subset_label" "$file"
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
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
      expected=(
        "results/$protocol/$model/$split/$subset/naturalness_scores.csv"
        "results/$protocol/$model/$split/$subset/naturalness_predictions_raw.csv"
      )
      if [[ "$split" == "test" ]]; then
        expected+=("results/$protocol/$model/$split/$subset/naturalness_scores_normalized.csv")
      fi
      check_files "22" "$model" "$split/$subset" "${expected[@]}"
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


# 50_turn_taking_inference.sh: compute DualTurn FVAD turn-taking surprisal metrics.
turntaking_experiment="group4-dualturn-full-all6-fvad256"
for split in "${splits[@]}"; do
  for subset in "${subsets[@]}"; do
    for model in "${models[@]}"; do
      check_files "50" "$model" "$split/$subset" \
        "results/$protocol/$model/$split/$subset/turntaking.${turntaking_experiment}.csv"
    done
  done
done

# 61_generate_report.sh: generate short reports, per-model graphs, and detailed HTML reports.
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
  check_files "61" "$model" "reports" "${expected[@]}"
done

# 62_benchmark.sh: generate per-model benchmark CSVs, merged benchmark CSV, and LaTeX table.
expected=(
  "reports/$protocol/benchmark.csv"
  "reports/$protocol/benchmark.tex"
)
for model in "${models[@]}"; do
  expected+=("reports/$protocol/benchmark/$model.csv")
done
check_files "62" "all" "benchmark" "${expected[@]}"

# 63_generate_article_graphs.sh: generate article figures and companion tables.
check_files "63" "all" "graphs" \
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
  "graphs/stage8_article_turntaking_naturalness_histogram.png"

exit "$missing_any"
