#!/usr/bin/env bash

set -uo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"

snapshot_ref=${PADJECTIVE_MIHARA_SNAPSHOT_REF:-paper}
schema=${PADJECTIVE_MIHARA_SCHEMA:-padjective}
output_root=${PADJECTIVE_MIHARA_SWEEP_DIR:-build/mihara-scaling-sweep}
folds_text=${PADJECTIVE_MIHARA_FOLDS:-"0 1 2 3 4"}

# Split the default or explicitly supplied space-separated values deliberately.
read -r -a folds <<< "$folds_text"
if [[ $# -gt 0 ]]; then
    budgets=("$@")
else
    budgets=(64 128 256 512 768 1024)
fi

mkdir -p "$output_root"

run_fold() {
    local budget=$1
    local fold=$2
    local budget_dir="$output_root/tags-$budget"
    local log_file="$budget_dir/fold-$fold.log"
    local complete_file="$budget_dir/fold-$fold.complete"
    mkdir -p "$budget_dir"

    if [[ -f "$complete_file" ]]; then
        echo "Skipping completed budget=$budget fold=$fold"
        return 0
    fi

    echo "Starting budget=$budget fold=$fold at $(date --iso-8601=seconds)"
    nice -n 10 uv run -m padjective.taxonomy_mihara_comparison \
        --schema "$schema" \
        --snapshot-ref "$snapshot_ref" \
        --max-tags "$budget" \
        --feature-selection frequency_independent \
        --fold "$fold" \
        >"$log_file" 2>&1
    local status=$?
    if [[ $status -eq 0 ]]; then
        touch "$complete_file"
        echo "Completed budget=$budget fold=$fold at $(date --iso-8601=seconds)"
    else
        echo "Failed budget=$budget fold=$fold status=$status at $(date --iso-8601=seconds)"
    fi
    return "$status"
}

overall_status=0
for budget in "${budgets[@]}"; do
    echo "Launching tag budget $budget across folds: ${folds[*]}"
    pids=()
    for fold in "${folds[@]}"; do
        run_fold "$budget" "$fold" &
        pids+=("$!")
    done

    budget_status=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            budget_status=1
            overall_status=1
        fi
    done
    echo "Finished tag budget $budget with status $budget_status at $(date --iso-8601=seconds)"
done

exit "$overall_status"
