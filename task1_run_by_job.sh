#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash task1_run_by_job.sh <carla_job_id|latest|none> <run|camera-smoke|debug-imports|smoke-model|train-velocity-head|official-route-smoke> [args...]"
    exit 2
fi

JOB_ID="$1"
MODE="$2"
shift 2

PROJECT_DIR="${PROJECT_DIR:-/lab/haoq_lab/cse12312032/simlingo}"
CARLA_RUNTIME_ROOT="${CARLA_RUNTIME_ROOT:-$PROJECT_DIR/logs/carla_runtime}"
CARLA_LEGACY_RUNTIME_ROOT="${CARLA_LEGACY_RUNTIME_ROOT:-/lab/haoq_lab/cse12312032/OpenDriveVLA/logs/carla_runtime}"
CARLA_LEGACY_RUNTIME_ROOT_2="${CARLA_LEGACY_RUNTIME_ROOT_2:-/lab/haoq_lab/cse12312032/openvla/logs/carla_runtime}"
CARLA_ROOT="${CARLA_ROOT:-/lab/haoq_lab/cse12312032/CARLA_0.9.15}"
CARLA_PYTHON_EGG="${CARLA_PYTHON_EGG:-$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg}"

PYTHON_SITE_VERSION="${PYTHON_SITE_VERSION:-$(python - <<'PY'
import sys
print(f"python{sys.version_info.major}.{sys.version_info.minor}")
PY
)}"
CARLA_CLIENT_SOURCE_SITE="${CARLA_CLIENT_SOURCE_SITE:-${CONDA_PREFIX:-/lab/haoq_lab/cse12312032/miniconda3/envs/simlingo}/lib/$PYTHON_SITE_VERSION/site-packages}"
CARLA_CLIENT_FALLBACK_SOURCE_SITE="${CARLA_CLIENT_FALLBACK_SOURCE_SITE:-}"
CARLA_CLIENT_FALLBACK_SOURCE_SITE_2="${CARLA_CLIENT_FALLBACK_SOURCE_SITE_2:-}"
CARLA_CLIENT_COMPAT_PY="${CARLA_CLIENT_COMPAT_PY:-$PROJECT_DIR/.carla_client_py}"
CARLA_CLIENT_COMPAT_LIB="${CARLA_CLIENT_COMPAT_LIB:-$CARLA_CLIENT_COMPAT_PY/carla.libs}"
CARLA_IMPORT_COMPAT_ROOT="${CARLA_IMPORT_COMPAT_ROOT:-$PROJECT_DIR/.carla_import_compat}"

find_connection_env() {
    local job_id="$1"
    local root="$2"
    if [ ! -d "$root" ]; then
        return 0
    fi
    if [ "$job_id" = "latest" ]; then
        find "$root" -mindepth 2 -maxdepth 2 -name connection.env -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr \
            | head -1 \
            | cut -d' ' -f2-
        return
    fi
    find "$root" -mindepth 2 -maxdepth 2 -path "*_job${job_id}/connection.env" -print 2>/dev/null \
        | sort \
        | tail -1
}

setup_pythonpath() {
    mkdir -p "$CARLA_CLIENT_COMPAT_PY"
    mkdir -p "$CARLA_IMPORT_COMPAT_ROOT"
    ln -sfn "$CARLA_ROOT" "$CARLA_IMPORT_COMPAT_ROOT/Carla"

    local compat_libcarla
    compat_libcarla="$(find "$CARLA_CLIENT_COMPAT_PY/carla" -maxdepth 1 -name 'libcarla*.so' -print -quit 2>/dev/null || true)"
    if [ -n "$compat_libcarla" ] && [[ "$compat_libcarla" != *"$PYTHON_SITE_VERSION"* ]]; then
        rm -f "$CARLA_CLIENT_COMPAT_PY/carla" "$CARLA_CLIENT_COMPAT_PY/carla.libs"
    fi

    if [ -d "$CARLA_CLIENT_COMPAT_PY/carla" ] \
        && [ -d "$CARLA_CLIENT_COMPAT_PY/carla.libs" ] \
        && find "$CARLA_CLIENT_COMPAT_PY/carla" -maxdepth 1 -name 'libcarla*.so' -print -quit | grep -q .; then
        CARLA_CLIENT_PYTHONPATH="$CARLA_CLIENT_COMPAT_PY"
    elif [ -d "$CARLA_CLIENT_SOURCE_SITE/carla" ] \
        && [ -d "$CARLA_CLIENT_SOURCE_SITE/carla.libs" ] \
        && find "$CARLA_CLIENT_SOURCE_SITE/carla" -maxdepth 1 -name 'libcarla*.so' -print -quit | grep -q .; then
        rm -f "$CARLA_CLIENT_COMPAT_PY/carla" "$CARLA_CLIENT_COMPAT_PY/carla.libs"
        ln -sfn "$CARLA_CLIENT_SOURCE_SITE/carla" "$CARLA_CLIENT_COMPAT_PY/carla"
        ln -sfn "$CARLA_CLIENT_SOURCE_SITE/carla.libs" "$CARLA_CLIENT_COMPAT_PY/carla.libs"
        CARLA_CLIENT_PYTHONPATH="$CARLA_CLIENT_COMPAT_PY"
    elif [ -n "$CARLA_CLIENT_FALLBACK_SOURCE_SITE" ] \
        && [ -d "$CARLA_CLIENT_FALLBACK_SOURCE_SITE/carla" ] \
        && [ -d "$CARLA_CLIENT_FALLBACK_SOURCE_SITE/carla.libs" ] \
        && find "$CARLA_CLIENT_FALLBACK_SOURCE_SITE/carla" -maxdepth 1 -name 'libcarla*.so' -print -quit | grep -q .; then
        rm -f "$CARLA_CLIENT_COMPAT_PY/carla" "$CARLA_CLIENT_COMPAT_PY/carla.libs"
        ln -sfn "$CARLA_CLIENT_FALLBACK_SOURCE_SITE/carla" "$CARLA_CLIENT_COMPAT_PY/carla"
        ln -sfn "$CARLA_CLIENT_FALLBACK_SOURCE_SITE/carla.libs" "$CARLA_CLIENT_COMPAT_PY/carla.libs"
        CARLA_CLIENT_PYTHONPATH="$CARLA_CLIENT_COMPAT_PY"
    elif [ -n "$CARLA_CLIENT_FALLBACK_SOURCE_SITE_2" ] \
        && [ -d "$CARLA_CLIENT_FALLBACK_SOURCE_SITE_2/carla" ] \
        && [ -d "$CARLA_CLIENT_FALLBACK_SOURCE_SITE_2/carla.libs" ] \
        && find "$CARLA_CLIENT_FALLBACK_SOURCE_SITE_2/carla" -maxdepth 1 -name 'libcarla*.so' -print -quit | grep -q .; then
        rm -f "$CARLA_CLIENT_COMPAT_PY/carla" "$CARLA_CLIENT_COMPAT_PY/carla.libs"
        ln -sfn "$CARLA_CLIENT_FALLBACK_SOURCE_SITE_2/carla" "$CARLA_CLIENT_COMPAT_PY/carla"
        ln -sfn "$CARLA_CLIENT_FALLBACK_SOURCE_SITE_2/carla.libs" "$CARLA_CLIENT_COMPAT_PY/carla.libs"
        CARLA_CLIENT_PYTHONPATH="$CARLA_CLIENT_COMPAT_PY"
    else
        if [ "$PYTHON_SITE_VERSION" != "python3.7" ] || [ ! -e "$CARLA_PYTHON_EGG" ]; then
            echo "ERROR: compatible CARLA Python package was not found for $PYTHON_SITE_VERSION."
            echo "Expected source: $CARLA_CLIENT_SOURCE_SITE/carla"
            echo "Expected libcarla: $CARLA_CLIENT_SOURCE_SITE/carla/libcarla*.so"
            echo "Missing fallback egg: $CARLA_PYTHON_EGG"
            echo "Install a CARLA wheel matching the active Python, e.g. carla==0.9.15 for this env."
            exit 2
        fi
        CARLA_CLIENT_PYTHONPATH="$CARLA_ROOT/PythonAPI:$CARLA_PYTHON_EGG"
    fi

    export CARLA_ROOT
    export CARLA_IMPORT_COMPAT_ROOT
    export WORK_DIR="$PROJECT_DIR"
    export SCENARIO_RUNNER_ROOT="$PROJECT_DIR/Bench2Drive/scenario_runner"
    export LEADERBOARD_ROOT="$PROJECT_DIR/Bench2Drive/leaderboard"
    export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/team_code:$PROJECT_DIR/Bench2Drive/leaderboard:$PROJECT_DIR/Bench2Drive/scenario_runner:$PROJECT_DIR/leaderboard:$PROJECT_DIR/scenario_runner:$CARLA_IMPORT_COMPAT_ROOT:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_CLIENT_PYTHONPATH:${PYTHONPATH:-}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:$CARLA_CLIENT_COMPAT_LIB:${LD_LIBRARY_PATH:-}"
    export SAVE_PATH="${SAVE_PATH:-$PROJECT_DIR/logs/simlingo_agent_debug}"
    export ROUTES="${ROUTES:-data/benchmarks/task1/task1.xml}"
}

setup_pythonpath
cd "$PROJECT_DIR"

case "$MODE" in
    debug-imports)
        python task_carla_code/task1_run_simlingo.py --mode debug-imports "$@"
        exit $?
        ;;
    smoke-model)
        python task_carla_code/task1_run_simlingo.py --mode smoke-model "$@"
        exit $?
        ;;
    train-velocity-head)
        python task_carla_code/task1_train_velocity_head.py "$@"
        exit $?
        ;;
esac

if [ "$JOB_ID" = "none" ]; then
    echo "ERROR: mode '$MODE' requires a CARLA job id or 'latest'."
    exit 2
fi

CONNECTION_ENV="$(find_connection_env "$JOB_ID" "$CARLA_RUNTIME_ROOT")"
if [ -z "$CONNECTION_ENV" ] || [ ! -f "$CONNECTION_ENV" ]; then
    CONNECTION_ENV="$(find_connection_env "$JOB_ID" "$CARLA_LEGACY_RUNTIME_ROOT")"
fi
if [ -z "$CONNECTION_ENV" ] || [ ! -f "$CONNECTION_ENV" ]; then
    CONNECTION_ENV="$(find_connection_env "$JOB_ID" "$CARLA_LEGACY_RUNTIME_ROOT_2")"
fi
if [ -z "$CONNECTION_ENV" ] || [ ! -f "$CONNECTION_ENV" ]; then
    echo "ERROR: could not find CARLA connection.env for job '$JOB_ID'."
    echo "Looked under:"
    echo "  $CARLA_RUNTIME_ROOT"
    echo "  $CARLA_LEGACY_RUNTIME_ROOT"
    echo "  $CARLA_LEGACY_RUNTIME_ROOT_2"
    exit 2
fi

source "$CONNECTION_ENV"
export CARLA_CONNECTION_ENV="$CONNECTION_ENV"

echo "=== CARLA connection selected ==="
echo "Job: ${CARLA_JOB_ID:-$JOB_ID}"
echo "Host: ${CARLA_HOST:-unknown}"
echo "Port: ${CARLA_PORT:-unknown}"
echo "Connection file: $CARLA_CONNECTION_ENV"
echo "Logs: ${CARLA_LOG_DIR:-unknown}"
echo ""

case "$MODE" in
    run)
        python task_carla_code/task1_run_simlingo.py \
            --mode run \
            --connection-env "$CARLA_CONNECTION_ENV" \
            "$@"
        ;;
    camera-smoke)
        python task_carla_code/task1_run_simlingo.py \
            --mode camera-smoke \
            --connection-env "$CARLA_CONNECTION_ENV" \
            "$@"
        ;;
    official-route-smoke)
        CHECKPOINT="${SIMLINGO_CHECKPOINT:-outputs/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt}"
        ROUTE="${SIMLINGO_ROUTE:-$PROJECT_DIR/leaderboard/data/bench2drive_split/bench2drive_00.xml}"
        RESULT="${SIMLINGO_RESULT:-$PROJECT_DIR/logs/task1/official_route_smoke/result.json}"
        mkdir -p "$(dirname "$RESULT")"
        python -u "$PROJECT_DIR/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py" \
            --routes="$ROUTE" \
            --repetitions=1 \
            --track=SENSORS \
            --checkpoint="$RESULT" \
            --timeout=600 \
            --agent="$PROJECT_DIR/team_code/agent_simlingo.py" \
            --agent-config="$CHECKPOINT" \
            --traffic-manager-seed=1 \
            --port="${CARLA_PORT:-2000}" \
            --traffic-manager-port="${CARLA_TM_PORT:-8000}"
        ;;
    *)
        echo "ERROR: unknown mode '$MODE'."
        exit 2
        ;;
esac
