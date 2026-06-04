#!/usr/bin/env bash

# Shared Slurm command builders for benchmark scripts.
# Usage: $(python_cmd SB10-S1 --gpu) bin/base_metrics.py --arg value

# exclude="--exclude=c18,c19,c21,octopod"
exclude='--exclude=octopod'

gpu_cmd() {
    printf "srun -p gpu --gpus 1 %s" "$exclude"
}
gpu_a100_cmd() {
    printf "srun -p gpu-a100 --account=a100acct --gpus 1 %s" "$exclude"
}
cpu_cmd() {
    printf "srun -p cpu --cpus-per-task 4 %s" "$exclude"
}

srun_cmd() {
    if [[ $# -ne 2 ]]; then
        echo "Usage: srun_cmd JOB_NAME --cpu|--gpu" >&2
        return 2
    fi

    local job_name="$1"
    local flag="$2"
    local base_cmd

    case "$flag" in
        --cpu)
            base_cmd="$(cpu_cmd)"
            ;;
        --gpu)
            base_cmd="$(gpu_cmd)"
            ;;
        --gpu-a100)
             base_cmd="$(gpu_a100_cmd)"
             ;;

        *)
            echo "Unknown execution flag: $flag. Expected --cpu or --gpu." >&2
            return 2
            ;;
    esac

    printf "%s --job-name %q" "$base_cmd" "$job_name"
}

python_cmd() {
    printf "%s python3" "$(srun_cmd "$@")"
}
