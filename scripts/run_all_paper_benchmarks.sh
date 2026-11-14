#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"

run_label=${PADJECTIVE_PAPER_RUN_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)}
output_root=${PADJECTIVE_PAPER_RUN_DIR:-build/paper-benchmark-reruns/$run_label}
database=${PADJECTIVE_PAPER_DATABASE:-shopifystores}
snapshot_ref=${PADJECTIVE_PAPER_SNAPSHOT_REF:-paper}
history_through=${PADJECTIVE_PAPER_HISTORY_THROUGH:-2026-09-02}
mihara_budgets_text=${PADJECTIVE_PAPER_MIHARA_BUDGETS:-"64 128 256 512 768 1024 1536 2048 2971"}
read -r -a mihara_budgets <<< "$mihara_budgets_text"

if [[ ! "$snapshot_ref" =~ ^[A-Za-z0-9._:-]+$ ]]; then
    echo "Unsafe snapshot reference: $snapshot_ref" >&2
    exit 2
fi
if [[ ! "$history_through" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "Invalid history cutoff: $history_through" >&2
    exit 2
fi

mkdir -p "$output_root" "$output_root/logs" "$output_root/complete" \
    "$output_root/before" "$output_root/after"

run_step() {
    local name=$1
    shift
    local marker="$output_root/complete/$name"
    local log="$output_root/logs/$name.log"
    if [[ -f "$marker" ]]; then
        echo "Skipping completed step: $name"
        return 0
    fi
    echo "Starting $name at $(date --iso-8601=seconds)"
    "$@" >"$log" 2>&1
    touch "$marker"
    echo "Completed $name at $(date --iso-8601=seconds)"
}

capture_table() {
    local phase=$1
    local table=$2
    psql -d "$database" --csv -c "SELECT * FROM padjective.$table" \
        > "$output_root/$phase/$table.csv"
}

capture_results() {
    local phase=$1
    local tables=(
        paper_revision_constrained_neural
        paper_revision_examples
        paper_revision_matched_budget_classical
        paper_revision_neural_sizes
        paper_revision_order_history
        paper_revision_q_sensitivity
        paper_revision_zubarev
        tag_network_analysis_runs
        taxonomy_mihara_fold_results
        taxonomy_mihara_runs
        umllr_order_ablation_fold_metrics
    )
    local table
    for table in "${tables[@]}"; do
        capture_table "$phase" "$table"
    done
}

capture_manifest() {
    local snapshot_id
    snapshot_id=$(psql -d "$database" -Atc \
        "SELECT snapshot_id FROM padjective.product_taxonomy_bench_snapshot_aliases WHERE alias = '$snapshot_ref'")
    if [[ -z "$snapshot_id" ]]; then
        echo "Snapshot alias not found: $snapshot_ref" >&2
        exit 3
    fi
    {
        echo "run_label=$run_label"
        echo "started_at_utc=$(date -u --iso-8601=seconds)"
        echo "host=$(hostname)"
        echo "git_commit=$(git rev-parse HEAD)"
        echo "snapshot_ref=$snapshot_ref"
        echo "snapshot_id=$snapshot_id"
        echo "history_through=$history_through"
        echo "mihara_budgets=$mihara_budgets_text"
        echo "python=$(uv run python --version 2>&1)"
        echo "uv=$(uv --version)"
    } > "$output_root/manifest.txt"
    psql -d "$database" --csv > "$output_root/before/paper_snapshot.csv" <<SQL
SELECT s.*
FROM padjective.product_taxonomy_bench_snapshots AS s
WHERE s.snapshot_id = '$snapshot_id';
SQL
    psql -d "$database" --csv > "$output_root/before/paper_fold_counts.csv" <<SQL
SELECT cv_fold, COUNT(*) AS product_count
FROM padjective.product_taxonomy_bench_products
WHERE snapshot_id = '$snapshot_id'
GROUP BY cv_fold
ORDER BY cv_fold NULLS FIRST;
SQL
}

backup_results() {
    local tables=(
        padjective.paper_revision_constrained_neural
        padjective.paper_revision_examples
        padjective.paper_revision_matched_budget_classical
        padjective.paper_revision_neural_sizes
        padjective.paper_revision_order_history
        padjective.paper_revision_q_sensitivity
        padjective.paper_revision_zubarev
        padjective.tag_network_analysis_runs
        padjective.taxonomy_mihara_coefficients
        padjective.taxonomy_mihara_fold_results
        padjective.taxonomy_mihara_predictions
        padjective.taxonomy_mihara_runs
        padjective.umllr_order_ablation_fold_metrics
        padjective.umllr_order_ablation_predictions
    )
    local dump_args=()
    local table
    for table in "${tables[@]}"; do
        dump_args+=(--table="$table")
    done
    pg_dump -d "$database" --data-only --no-owner --no-privileges \
        "${dump_args[@]}" | gzip -9 > "$output_root/before/results.sql.gz"
}

run_neural_fold() {
    local fold=$1
    run_step "neural_fold_$fold" nice -n 10 uv run -m \
        padjective.paper_revision_experiments \
        --snapshot-ref "$snapshot_ref" \
        --hidden-sizes 4,8,12,24,48,2000 \
        --neural-folds "$fold" \
        --skip-zubarev --skip-q
}

run_step manifest capture_manifest
run_step before_results capture_results before
run_step backup_results backup_results
run_step tests uv run -m pytest -q
run_step snapshot_export uv run -m padjective.product_taxonomy_bench_export \
    --snapshot "$snapshot_ref" \
    --out-root "$output_root/snapshot" \
    --formats jsonl --no-gzip
run_step benchmark_bundle nice -n 10 uv run -m padjective.benchmark_bundle \
    --snapshot-dir "$output_root/snapshot/$snapshot_ref" \
    --out-dir "$output_root/benchmark_bundle"
run_step umllr_ablation nice -n 10 uv run -m padjective.umllr \
    --snapshot-ref "$snapshot_ref" \
    --tag-order-strategy all --ablation-only

background_pids=()
for fold in 0 1 2 3 4; do
    run_neural_fold "$fold" &
    background_pids+=("$!")
done
run_step revision_controls nice -n 10 uv run -m \
    padjective.paper_revision_experiments \
    --snapshot-ref "$snapshot_ref" \
    --run-matched-budget-classical --skip-neural &
background_pids+=("$!")
run_step examples nice -n 10 uv run -m padjective.paper_revision_examples \
    --snapshot-ref "$snapshot_ref" &
background_pids+=("$!")
run_step tag_network nice -n 10 uv run -m padjective.tag_network_analysis \
    --snapshot-ref "$snapshot_ref" \
    --output "$output_root/tag_network_analysis.json" \
    --report-snapshot-output "$output_root/tag_network_report_snapshot.json" &
background_pids+=("$!")

background_status=0
for pid in "${background_pids[@]}"; do
    if ! wait "$pid"; then
        background_status=1
    fi
done
if [[ $background_status -ne 0 ]]; then
    echo "One or more fixed-snapshot benchmark jobs failed" >&2
    exit "$background_status"
fi

run_step neural_scaling uv run -m padjective.paper_revision_neural_scaling \
    --snapshot-ref "$snapshot_ref" \
    --output "$output_root/neural_scaling.eps" \
    --tex-output "$output_root/neural_scaling_stats.tex" \
    --figure-one-bundle "$output_root/benchmark_bundle/benchmark.json"

background_pids=()
run_step mihara_baseline nice -n 10 uv run -m \
    padjective.taxonomy_mihara_comparison \
    --snapshot-ref "$snapshot_ref" \
    --max-tags 32 --feature-selection frequency &
background_pids+=("$!")
PADJECTIVE_MIHARA_SWEEP_DIR="$output_root/mihara_sweep" \
PADJECTIVE_MIHARA_SNAPSHOT_REF="$snapshot_ref" \
    run_step mihara_sweep nice -n 10 \
    scripts/run_mihara_scaling_sweep.sh "${mihara_budgets[@]}" &
background_pids+=("$!")
run_step order_history nice -n 10 uv run -m padjective.paper_order_history \
    --force-rerun --through-date "$history_through" \
    --output-dir "$output_root/order_history" &
background_pids+=("$!")

background_status=0
for pid in "${background_pids[@]}"; do
    if ! wait "$pid"; then
        background_status=1
    fi
done
if [[ $background_status -ne 0 ]]; then
    echo "One or more historical or Mihara benchmark jobs failed" >&2
    exit "$background_status"
fi

run_step after_results capture_results after
find "$output_root" -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$output_root/SHA256SUMS"
echo "completed_at_utc=$(date -u --iso-8601=seconds)" >> "$output_root/manifest.txt"
echo "All paper benchmarks completed: $output_root"
