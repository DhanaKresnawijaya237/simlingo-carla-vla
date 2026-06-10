#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash task1_eval_20x_random.sh <carla_job_id|latest> [extra task1_run_by_job.sh args...]"
    exit 2
fi

JOB_ID="$1"
shift

CHECKPOINT="${CHECKPOINT:-outputs/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt}"
DEFAULT_VELOCITY_HEAD_CHECKPOINT="logs/task1_velocity_head/v8_relative_speed/best.pt"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SCENARIO_SEED="${SCENARIO_SEED:-$(date +%s)}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/task1/simlingo_real_eval_20x_${RUN_ID}}"
TRIALS_PER_COMMAND="${TRIALS_PER_COMMAND:-20}"
MIN_SPAWN_DISTANCE_M="${MIN_SPAWN_DISTANCE_M:-35}"
MAX_SPAWN_CANDIDATES="${MAX_SPAWN_CANDIDATES:-2000}"
SPAWN_DIVERSITY_SCOPE="${SPAWN_DIVERSITY_SCOPE:-command}"
MAX_OFFROAD_FRAMES="${MAX_OFFROAD_FRAMES:-5}"
if [ -z "${VELOCITY_HEAD_CHECKPOINT+x}" ] && [ -f "$DEFAULT_VELOCITY_HEAD_CHECKPOINT" ]; then
    VELOCITY_HEAD_CHECKPOINT="$DEFAULT_VELOCITY_HEAD_CHECKPOINT"
fi
VELOCITY_HEAD_ARGS=()
if [ -n "${VELOCITY_HEAD_CHECKPOINT:-}" ] && [ "$VELOCITY_HEAD_CHECKPOINT" != "none" ]; then
    VELOCITY_HEAD_ARGS+=(--velocity-head-checkpoint "$VELOCITY_HEAD_CHECKPOINT")
fi

echo "=== SimLingo Task 1 real eval ==="
echo "CARLA job: $JOB_ID"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Trials per command: $TRIALS_PER_COMMAND"
echo "Scenario seed: $SCENARIO_SEED"
echo "Min spawn distance: ${MIN_SPAWN_DISTANCE_M}m"
echo "Spawn diversity scope: $SPAWN_DIVERSITY_SCOPE"
echo "Max offroad frames: $MAX_OFFROAD_FRAMES"
if [ -n "${VELOCITY_HEAD_CHECKPOINT:-}" ] && [ "$VELOCITY_HEAD_CHECKPOINT" != "none" ]; then
    echo "Velocity head: $VELOCITY_HEAD_CHECKPOINT"
else
    echo "Velocity head: disabled"
fi
echo ""

bash task1_run_by_job.sh "$JOB_ID" run \
    --checkpoint "$CHECKPOINT" \
    --eval-suite basic \
    --trials-per-command "$TRIALS_PER_COMMAND" \
    --duration 20 \
    --eval-duration-overrides "turn left:35,turn right:35,straight:22,stop:22,speed up:28,slow down:28" \
    --fps 10 \
    --sim-fps 20 \
    --policy-hz 0.20 \
    --velocity-head-hz 5.0 \
    --simlingo-speed-control-mode model-speed-commands \
    --speed-command-route-source map \
    --camera-view topdown \
    --sensor-warmup-ticks 40 \
    --sensor-retry-ticks 10 \
    --spawn-policy junction \
    --scenario-seed "$SCENARIO_SEED" \
    --max-spawn-candidates "$MAX_SPAWN_CANDIDATES" \
    --min-spawn-distance-m "$MIN_SPAWN_DISTANCE_M" \
    --spawn-diversity-scope "$SPAWN_DIVERSITY_SCOPE" \
    --min-pre-junction-route-points 4 \
    --straight-require-junction \
    --speed-route-max-heading 8 \
    --speed-command-warmup-sec 5 \
    --stop-warmup-sec 5 \
    --speed-delta-threshold 0.8 \
    --slowdown-min-success-speed-mps 1.0 \
    --stop-speed-threshold 0.35 \
    --stop-baseline-min-speed-mps 1.0 \
    --speed-hold-sec 3 \
    --max-offroad-frames "$MAX_OFFROAD_FRAMES" \
    --output-dir "$OUTPUT_DIR" \
    "${VELOCITY_HEAD_ARGS[@]}" \
    "$@"

echo ""
echo "Done. Results:"
echo "  $OUTPUT_DIR/summary.json"
echo "  $OUTPUT_DIR/summary.csv"
echo "  $OUTPUT_DIR/failure_analysis.md"
