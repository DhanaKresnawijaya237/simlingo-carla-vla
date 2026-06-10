#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$#" -lt 2 ]; then
    cat <<'EOF'
Usage:
  bash task1_eval_multimap.sh <run_name> <map_label:carla_job_id> [<map_label:carla_job_id> ...] [-- extra task1_eval_20x_random.sh args]

Example:
  bash task1_eval_multimap.sh final_20x \
    Town01:89101 Town02:89102 Town03:89103 Town04:89104 Town05_Opt:89066

Outputs:
  logs/task1/<run_name>/<map_label>/summary.json
  logs/task1/<run_name>/summary.json
  logs/task1/<run_name>/summary.csv
  logs/task1/<run_name>/failure_analysis.md
EOF
    exit 2
fi

RUN_NAME="$1"
shift

MAP_JOBS=()
EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--" ]; then
        shift
        EXTRA_ARGS=("$@")
        break
    fi
    MAP_JOBS+=("$1")
    shift
done

ROOT_DIR="${ROOT_DIR:-logs/task1/${RUN_NAME}}"
TARGET_SUCCESS_RATE="${TARGET_SUCCESS_RATE:-0.80}"
mkdir -p "$ROOT_DIR"

echo "=== SimLingo Task 1 multi-map eval ==="
echo "Run name: $RUN_NAME"
echo "Root: $ROOT_DIR"
echo "Maps: ${#MAP_JOBS[@]}"
echo ""

STATUSES=()
for pair in "${MAP_JOBS[@]}"; do
    if [[ "$pair" != *:* ]]; then
        echo "ERROR: expected map_label:carla_job_id, got '$pair'" >&2
        STATUSES+=("2")
        continue
    fi
    MAP_LABEL="${pair%%:*}"
    JOB_ID="${pair#*:}"
    MAP_OUTPUT="${ROOT_DIR}/${MAP_LABEL}"
    echo "=== Running map $MAP_LABEL with CARLA job $JOB_ID ==="
    OUTPUT_DIR="$MAP_OUTPUT" RUN_ID="$RUN_NAME-$MAP_LABEL" bash task1_eval_20x_random.sh "$JOB_ID" "${EXTRA_ARGS[@]}"
    STATUS="$?"
    STATUSES+=("$STATUS")
    if [ "$STATUS" -ne 0 ]; then
        echo "WARNING: map $MAP_LABEL exited with status $STATUS; continuing so aggregate results are still written."
    fi
    echo ""
done

echo "=== Aggregating multi-map results ==="
python task_carla_code/task1_aggregate_multimap.py \
    --root "$ROOT_DIR" \
    --target-success-rate "$TARGET_SUCCESS_RATE"
AGG_STATUS="$?"

echo ""
echo "Done. Aggregate results:"
echo "  $ROOT_DIR/summary.json"
echo "  $ROOT_DIR/summary.csv"
echo "  $ROOT_DIR/failure_analysis.md"

exit "$AGG_STATUS"
