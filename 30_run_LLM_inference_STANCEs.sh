#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source openai_keys.sh

INDICES=(0 1 2 3 4 5 6 7 8 9)
for split in 'test'; do
    for subset in 'improvised'; do
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
            srun -p cpu \
                python3 bin/STANCE/make_questions.py \
                --metadata "$metadata" \
                --questions-csv "$questions_csv" \
                --roles "${ROLES[@]}" \
                --input-mode "$INPUT_MODE" \
                --assets_dir "$seamless_assets_dir" \
                --question-index "$idx" 

            srun -p cpu \
                python3 bin/STANCE/score.py \
                --metadata "$metadata" \
                --questions-csv "$questions_csv" \
                --outputs_dir "$metrics" \
                --eval_model "$stance_llm_model" \
                --openai-api-key "$openai_api_key" \
                --openai-org "$org"

            done
        done
    done
done

  
