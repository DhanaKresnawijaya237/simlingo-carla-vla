#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash task1_eval_20x_random.sh <carla_job_id|latest> [extra task1_run_by_job.sh args...]"
    exit 2
fi

JOB_ID="$1"
shift

CHECKPOINT="${CHECKPOINT:-outputs/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SCENARIO_SEED="${SCENARIO_SEED:-$(date +%s)}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/task1/simlingo_real_eval_20x_${RUN_ID}}"
TRIALS_PER_COMMAND="${TRIALS_PER_COMMAND:-20}"
MIN_SPAWN_DISTANCE_M="${MIN_SPAWN_DISTANCE_M:-35}"
MAX_SPAWN_CANDIDATES="${MAX_SPAWN_CANDIDATES:-2000}"
SPAWN_DIVERSITY_SCOPE="${SPAWN_DIVERSITY_SCOPE:-command}"

echo "=== SimLingo Task 1 real eval ==="
echo "CARLA job: $JOB_ID"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Trials per command: $TRIALS_PER_COMMAND"
echo "Scenario seed: $SCENARIO_SEED"
echo "Min spawn distance: ${MIN_SPAWN_DISTANCE_M}m"
echo "Spawn diversity scope: $SPAWN_DIVERSITY_SCOPE"
echo ""

bash task1_run_by_job.sh "$JOB_ID" run \
    --checkpoint "$CHECKPOINT" \
    --eval-suite basic \
    --trials-per-command "$TRIALS_PER_COMMAND" \
    --duration 20 \
    --eval-duration-overrides "turn left:35,turn right:35,straight:14,stop:16,speed up:20,slow down:20" \
    --fps 10 \
    --sim-fps 20 \
    --policy-hz 0.20 \
    --camera-view topdown \
    --sensor-warmup-ticks 40 \
    --sensor-retry-ticks 10 \
    --spawn-policy junction \
    --scenario-seed "$SCENARIO_SEED" \
    --max-spawn-candidates "$MAX_SPAWN_CANDIDATES" \
    --min-spawn-distance-m "$MIN_SPAWN_DISTANCE_M" \
    --spawn-diversity-scope "$SPAWN_DIVERSITY_SCOPE" \
    --speed-route-max-heading 8 \
    --speed-command-warmup-sec 3 \
    --stop-warmup-sec 3 \
    --speed-delta-threshold 0.8 \
    --slowdown-min-success-speed-mps 1.0 \
    --stop-speed-threshold 0.35 \
    --stop-baseline-min-speed-mps 1.0 \
    --speed-hold-sec 2 \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo ""
echo "Done. Results:"
echo "  $OUTPUT_DIR/summary.json"
echo "  $OUTPUT_DIR/summary.csv"
echo "  $OUTPUT_DIR/failure_analysis.md"
