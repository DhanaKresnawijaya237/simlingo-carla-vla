#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$#" -lt 3 ]; then
    cat <<'EOF'
Usage:
  bash task1_eval_one_map.sh <run_name> <map_label> <carla_job_id|latest> [extra task1_eval_20x_random.sh args...]

Example:
  bash task1_eval_one_map.sh final_20x Town05_Opt 89066

This writes:
  logs/task1/<run_name>/<map_label>/summary.json

Then refreshes:
  logs/task1/<run_name>/summary.json
  logs/task1/<run_name>/summary.csv
  logs/task1/<run_name>/failure_analysis.md
EOF
    exit 2
fi

RUN_NAME="$1"
MAP_LABEL="$2"
JOB_ID="$3"
shift 3

ROOT_DIR="${ROOT_DIR:-logs/task1/${RUN_NAME}}"
MAP_OUTPUT="${ROOT_DIR}/${MAP_LABEL}"
TARGET_SUCCESS_RATE="${TARGET_SUCCESS_RATE:-0.80}"

mkdir -p "$ROOT_DIR"

echo "=== SimLingo Task 1 one-map eval ==="
echo "Run name: $RUN_NAME"
echo "Map label: $MAP_LABEL"
echo "CARLA job: $JOB_ID"
echo "Output: $MAP_OUTPUT"
echo ""

set +e
OUTPUT_DIR="$MAP_OUTPUT" RUN_ID="$RUN_NAME-$MAP_LABEL" bash task1_eval_20x_random.sh "$JOB_ID" "$@"
EVAL_STATUS="$?"
set -e
if [ "$EVAL_STATUS" -ne 0 ]; then
    echo "WARNING: map $MAP_LABEL exited with status $EVAL_STATUS; refreshing aggregate results anyway."
fi

echo ""
echo "=== Refreshing aggregate results ==="
python task_carla_code/task1_aggregate_multimap.py \
    --root "$ROOT_DIR" \
    --target-success-rate "$TARGET_SUCCESS_RATE" || true

echo ""
echo "Done. Current aggregate results:"
echo "  $ROOT_DIR/summary.json"
echo "  $ROOT_DIR/summary.csv"
echo "  $ROOT_DIR/failure_analysis.md"
