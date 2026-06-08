#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source config.sh
source cmd.sh

echo "Preparing data from seamless interactions for SPEARBench"
echo "Currently selects interactions with at least 2 turns, 2 speakers, ending by a question"
echo "Input data directory: $seamless_data_dir"
echo "Output data directory: $data_dir/'inputs' "

for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        $(python_cmd 'SB01' --cpu) bin/prepare_data_from_seamless.py \
            --data_dir $seamless_data_dir \
            --questions_output_dir $data_dir/'inputs' \
            --answers_output_dir $data_dir/'outputs/original' \
            --min_turns 2 \
            --min_speakers 2 \
            --method 'end_with_question' \
            --split $split \
            --subset $subset &

        sleep 1
        
    done
done

wait

for split in 'test' 'dev'; do
    for subset in 'improvised' 'naturalistic'; do
        python3 bin/data_prep_summary.py \
            --answers_output_dir $data_dir/outputs/original/${split}/${subset} \
            --original_data_csv $seamless_assets_dir/dyad_lookup.csv \
            --data_dir $seamless_data_dir \
            --split $split \
            --subset $subset
    done
done

echo "LaTeX data preparation summary:"
python3 bin/data_prep_summary.py \
    --latex-table \
    --answers_root $data_dir/outputs/original \
    --original_data_csv $seamless_assets_dir/dyad_lookup.csv \
    --data_dir $seamless_data_dir
