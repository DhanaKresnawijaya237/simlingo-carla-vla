#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$#" -lt 1 ]; then
    cat <<'EOF'
Usage:
  bash task1_smoke_speed_waypoints.sh <carla_job_id|latest> [extra task1_run_by_job.sh args...]

Purpose:
  Smoke-test raw SimLingo speed waypoints for:
    stop, speed up, slow down

  This disables the Task 1 velocity head and uses:
    SimLingo predicted speed waypoints -> scalar target speed

Example:
  bash task1_smoke_speed_waypoints.sh "$JOB_ID"

Useful overrides:
  TRIALS_PER_COMMAND=5 OUTPUT_DIR=logs/task1/speed_waypoint_smoke_v2 \
    bash task1_smoke_speed_waypoints.sh "$JOB_ID"
EOF
    exit 2
fi

JOB_ID="$1"
shift

CHECKPOINT="${CHECKPOINT:-outputs/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/task1/speed_waypoint_smoke_${RUN_ID}}"
TRIALS_PER_COMMAND="${TRIALS_PER_COMMAND:-3}"
SCENARIO_SEED="${SCENARIO_SEED:-12345}"
POLICY_HZ="${POLICY_HZ:-0.40}"
SPEED_COMMAND_WARMUP_SEC="${SPEED_COMMAND_WARMUP_SEC:-5}"
STOP_WARMUP_SEC="${STOP_WARMUP_SEC:-5}"
SPEED_HOLD_SEC="${SPEED_HOLD_SEC:-3}"

echo "=== SimLingo raw speed-waypoint smoke ==="
echo "CARLA job: $JOB_ID"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Commands: stop, speed up, slow down"
echo "Trials per command: $TRIALS_PER_COMMAND"
echo "Policy Hz: $POLICY_HZ"
echo "Warmup: speed=${SPEED_COMMAND_WARMUP_SEC}s stop=${STOP_WARMUP_SEC}s"
echo "Velocity head: disabled"
echo "Speed source under command: raw SimLingo model speed waypoints"
echo ""

VELOCITY_HEAD_CHECKPOINT=none bash task1_run_by_job.sh "$JOB_ID" run \
    --checkpoint "$CHECKPOINT" \
    --eval-suite basic \
    --eval-commands "stop,speed up,slow down" \
    --trials-per-command "$TRIALS_PER_COMMAND" \
    --duration 28 \
    --eval-duration-overrides "stop:22,speed up:28,slow down:28" \
    --fps 10 \
    --sim-fps 20 \
    --policy-hz "$POLICY_HZ" \
    --simlingo-input-mode strict-command \
    --simlingo-execution-mode cached-route-follower \
    --simlingo-speed-control-mode model \
    --speed-command-route-source model \
    --camera-view topdown \
    --sensor-warmup-ticks 40 \
    --sensor-retry-ticks 10 \
    --spawn-policy junction \
    --scenario-seed "$SCENARIO_SEED" \
    --speed-route-max-heading 8 \
    --speed-command-warmup-sec "$SPEED_COMMAND_WARMUP_SEC" \
    --stop-warmup-sec "$STOP_WARMUP_SEC" \
    --speed-delta-threshold 0.8 \
    --slowdown-min-success-speed-mps 1.0 \
    --stop-speed-threshold 0.35 \
    --stop-baseline-min-speed-mps 1.0 \
    --speed-hold-sec "$SPEED_HOLD_SEC" \
    --max-offroad-frames 5 \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo ""
echo "Done. Check:"
echo "  $OUTPUT_DIR/summary.json"
echo "  $OUTPUT_DIR/summary.csv"
echo "  $OUTPUT_DIR/failure_analysis.md"
echo ""
echo "Expected tick-log evidence during active commands:"
echo "  cached_follow.cached_speed_source = model"
echo "  cached_follow.cached_velocity_head_target_speed_mps = null"
