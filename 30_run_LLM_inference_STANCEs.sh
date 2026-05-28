#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source openai_keys.sh

run_question=0
run_inference=0
force_recompute=0

if [[ $# -eq 0 ]]; then
  run_question=1
  run_inference=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --question|--stage1)
      run_question=1
      ;;
    --inference|--stage2)
      run_inference=1
      ;;
    --all)
      run_question=1
      run_inference=1
      ;;
    --force-recompute)
      force_recompute=1
      ;;
    -h|--help)
      cat <<EOF
Usage: bash 30_run_LLM_inference_STANCEs.sh [--question] [--inference] [--all] [--force-recompute]

Stages:
  --question         Stage 1: create STANCE question CSVs
  --inference        Stage 2: score STANCE question CSVs with the LLM judge
  --all              Run both stages
  --force-recompute  Stage 2 recomputes all valid rows instead of only failed/missing rows

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

INDICES=(0 1 2 3 4 5 6 7 8 9)
force_args=()
if [[ "$force_recompute" -eq 1 ]]; then
  force_args+=(--force-recompute)
fi

if [[ "$run_question" -eq 1 ]]; then
for split in 'test' 'dev'; do
    for subset in 'improvised'; do #never run on naturalistic, original paper needs ground truth stances, only available for the improvised subset
        for model in 'original' $llm_model; do
            metadata="data/$protocol/outputs/$model/$split/$subset/metadata.csv"
            metrics="results/$protocol/$model/$split/$subset"
            for idx in "${INDICES[@]}"; do
            # ----- Roles-of-interest for THIS question index -----
            # These must correspond to roles whose category_* matches one of the
            # selected question's related_categories.
            case "${idx}" in
                0)
                ROLES=(Friendly Warm Approachable Welcoming Aloof Distant Impersonal Indifferent)
                INPUT_MODE="single"
                ;;
                1)
                ROLES=(Empathetic Considerate Understanding Concerned Insensitive Unsympathetic Inconsiderate Callous)
                INPUT_MODE="single"
                ;;
                2)
                ROLES=(Polite Respectful Courteous Disrespectful Impolite Uncivil)
                INPUT_MODE="single"
                ;;
                3)
                ROLES=(Assertive Decisive Self-assured Firm Indecisive Self-doubting Unassertive Timid)
                INPUT_MODE="single"
                ;;
                4)
                ROLES=(Honest Ingenuous Uncalculating Manipulative Calculating Devious)
                INPUT_MODE="single"
                ;;
                5)
                ROLES=(Alert Attentive Concentrating Engaged Bewildered Distracted Drowsy Unfocused)
                INPUT_MODE="single"
                ;;
                6)
                ROLES=(Goal-oriented Driven Self-disciplined Organized Unmotivated Disorganized Inconsistent Unproductive)
                INPUT_MODE="single"
                ;;
                7)
                ROLES=(Engaging Sociable Gregarious Outgoing Withdrawn Disengaged Reticent Taciturn)
                INPUT_MODE="interaction"
                ;;
                8)
                ROLES=(Submissive Meek Yielding Undemanding Forceful Overbearing Dominant Domineering)
                INPUT_MODE="interaction"
                ;;
                9)
                ROLES=(Stable Steady Unaggressive Unargumentative Aggressive Cruel Ruthless Vindictive)
                INPUT_MODE="interaction"
                ;;
                *)
                echo "[ERROR] No ROLES configured for index=${idx}" >&2
                exit 2
                ;;
            esac

            echo "Predict STANCE Q$idx outputs for split:$split subset:$subset model:$model"
            questions_csv="data/$protocol/outputs/$model/$split/$subset/stance_questions_Q${idx}.csv"
            
            python3 bin/STANCE/make_questions.py \
            --metadata "$metadata" \
            --questions-csv "$questions_csv" \
            --roles "${ROLES[@]}" \
            --input-mode "$INPUT_MODE" \
            --assets_dir "$seamless_assets_dir" \
            --question-index "$idx" &

            done
        done
    done
done
wait
echo "All STANCE questions generated for original and $llm_model outputs."
fi

if [[ "$run_inference" -eq 1 ]]; then
for split in 'test' 'dev'; do
    for subset in 'improvised'; do #never run on naturalistic, original paper needs ground truth stances, only available for the improvised subset
        for model in 'original' $llm_model; do
            metadata="data/$protocol/outputs/$model/$split/$subset/metadata.csv"
            metrics="results/$protocol/$model/$split/$subset"
            for idx in "${INDICES[@]}"; do
            # ----- Roles-of-interest for THIS question index -----
            # These must correspond to roles whose category_* matches one of the
            # selected question's related_categories.
            case "${idx}" in
                0)
                ROLES=(Friendly Warm Approachable Welcoming Aloof Distant Impersonal Indifferent)
                INPUT_MODE="single"
                ;;
                1)
                ROLES=(Empathetic Considerate Understanding Concerned Insensitive Unsympathetic Inconsiderate Callous)
                INPUT_MODE="single"
                ;;
                2)
                ROLES=(Polite Respectful Courteous Disrespectful Impolite Uncivil)
                INPUT_MODE="single"
                ;;
                3)
                ROLES=(Assertive Decisive Self-assured Firm Indecisive Self-doubting Unassertive Timid)
                INPUT_MODE="single"
                ;;
                4)
                ROLES=(Honest Ingenuous Uncalculating Manipulative Calculating Devious)
                INPUT_MODE="single"
                ;;
                5)
                ROLES=(Alert Attentive Concentrating Engaged Bewildered Distracted Drowsy Unfocused)
                INPUT_MODE="single"
                ;;
                6)
                ROLES=(Goal-oriented Driven Self-disciplined Organized Unmotivated Disorganized Inconsistent Unproductive)
                INPUT_MODE="single"
                ;;
                7)
                ROLES=(Engaging Sociable Gregarious Outgoing Withdrawn Disengaged Reticent Taciturn)
                INPUT_MODE="interaction"
                ;;
                8)
                ROLES=(Submissive Meek Yielding Undemanding Forceful Overbearing Dominant Domineering)
                INPUT_MODE="interaction"
                ;;
                9)
                ROLES=(Stable Steady Unaggressive Unargumentative Aggressive Cruel Ruthless Vindictive)
                INPUT_MODE="interaction"
                ;;
                *)
                echo "[ERROR] No ROLES configured for index=${idx}" >&2
                exit 2
                ;;
            esac

            
            questions_csv="data/$protocol/outputs/$model/$split/$subset/stance_questions_Q${idx}.csv"
            job_name="SB30-id$idx"
            if [[ -f "$metrics/stance_metrics_Q${idx}.csv" ]]; then #ignore existing non-empty files unless --force-recompute is set
              continue
            fi
            echo "Predict STANCE Q$idx outputs for split:$split subset:$subset model:$model"
            $(python_cmd "$job_name" --cpu) bin/STANCE/score.py \
                --metadata "$metadata" \
                --questions-csv "$questions_csv" \
                --outputs_dir "$metrics" \
                --eval_model "$stance_llm_model" \
                --openai-api-key "$openai_api_key" \
                --openai-org "$org" \
                "${force_args[@]}" &

            done
        done
    done
done
wait
echo "All STANCE questions scored for original and $llm_model outputs."
fi

exit
  
