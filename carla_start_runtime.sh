#!/usr/bin/env bash
# Long-running CARLA server for the SimLingo Task 1 workflow.
#
# Submit:
#   cd /lab/haoq_lab/cse12312032/simlingo
#   sbatch --export=ALL,CARLA_MAP=Town10HD_Opt carla_start_runtime.sh
#
# New runtime connection files are written under:
#   simlingo/logs/carla_runtime/<timestamp>_job<job_id>/connection.env

#SBATCH -J carla_runtime
#SBATCH -p a100
#SBATCH --qos=a100
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lab/haoq_lab/cse12312032/simlingo}"
CARLA_DIR="${CARLA_DIR:-/lab/haoq_lab/cse12312032/CARLA_0.9.15}"
CONDA_BIN="${CONDA_BIN:-/lab/haoq_lab/cse12312032/miniconda3/bin/conda}"
CARLA_PYTHON_CONDA_ENV="${CARLA_PYTHON_CONDA_ENV:-simlingo}"
CARLA_ROOT="${CARLA_ROOT:-$CARLA_DIR}"
CARLA_PYTHON_EGG="${CARLA_PYTHON_EGG:-$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg}"
CARLA_CLIENT_SOURCE_SITE="${CARLA_CLIENT_SOURCE_SITE:-/lab/haoq_lab/cse12312032/miniconda3/envs/simlingo/lib/python3.8/site-packages}"
CARLA_CLIENT_COMPAT_PY="${CARLA_CLIENT_COMPAT_PY:-$PROJECT_DIR/.carla_client_py}"
CARLA_CLIENT_COMPAT_LIB="${CARLA_CLIENT_COMPAT_LIB:-$CARLA_CLIENT_COMPAT_PY/carla.libs}"

RUNTIME_ASSET_DIR="${CARLA_RUNTIME_ASSET_DIR:-$PROJECT_DIR/runtime}"
CARLA_SIF="${CARLA_SIF:-$RUNTIME_ASSET_DIR/carla_0.9.15.sif}"
NVIDIA_DRIVER_LIBS_ROOT="${NVIDIA_DRIVER_LIBS_ROOT:-$RUNTIME_ASSET_DIR/nvidia-driver-libs}"
LEGACY_CARLA_SIF="${LEGACY_CARLA_SIF:-/lab/haoq_lab/cse12312032/openvla/project/carla_0.9.15.sif}"
LEGACY_NVIDIA_DRIVER_LIBS_ROOT="${LEGACY_NVIDIA_DRIVER_LIBS_ROOT:-/lab/haoq_lab/cse12312032/openvla/nvidia-driver-libs}"

RPC_PORT="${CARLA_RPC_PORT:-2000}"
CARLA_MAP="${CARLA_MAP:-}"
LOG_DIR="${CARLA_RUNTIME_ROOT:-$PROJECT_DIR/logs/carla_runtime}"
STAMP="$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-manual}"
RUN_DIR="$LOG_DIR/$STAMP"

CARLA_PID=""

cleanup() {
    if [ -n "$CARLA_PID" ] && kill -0 "$CARLA_PID" 2>/dev/null; then
        echo "=== Stopping CARLA PID $CARLA_PID ==="
        kill "$CARLA_PID" 2>/dev/null || true
        wait "$CARLA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

mkdir -p "$RUN_DIR"
cd "$PROJECT_DIR"

if command -v singularity >/dev/null 2>&1; then
    CONTAINER_CMD="singularity"
elif command -v apptainer >/dev/null 2>&1; then
    CONTAINER_CMD="apptainer"
else
    echo "ERROR: neither singularity nor apptainer is available."
    exit 2
fi

if [ ! -f "$CARLA_SIF" ] && [ -f "$LEGACY_CARLA_SIF" ]; then
    echo "WARNING: using legacy CARLA SIF: $LEGACY_CARLA_SIF"
    echo "Move or link it to: $RUNTIME_ASSET_DIR/carla_0.9.15.sif"
    CARLA_SIF="$LEGACY_CARLA_SIF"
fi
if [ ! -f "$CARLA_SIF" ]; then
    echo "ERROR: CARLA SIF not found: $CARLA_SIF"
    echo "Expected SimLingo-local asset: $RUNTIME_ASSET_DIR/carla_0.9.15.sif"
    exit 2
fi

if [ ! -d "$CARLA_DIR" ]; then
    echo "ERROR: CARLA_DIR not found: $CARLA_DIR"
    exit 2
fi

if [ ! -d "$NVIDIA_DRIVER_LIBS_ROOT" ] && [ -d "$LEGACY_NVIDIA_DRIVER_LIBS_ROOT" ]; then
    echo "WARNING: using legacy NVIDIA driver libraries: $LEGACY_NVIDIA_DRIVER_LIBS_ROOT"
    echo "Move or link them to: $RUNTIME_ASSET_DIR/nvidia-driver-libs"
    NVIDIA_DRIVER_LIBS_ROOT="$LEGACY_NVIDIA_DRIVER_LIBS_ROOT"
fi

DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
NVIDIA_LIB_DIR="$NVIDIA_DRIVER_LIBS_ROOT/$DRIVER_VERSION"
if [ ! -d "$NVIDIA_LIB_DIR" ]; then
    echo "ERROR: missing NVIDIA graphics lib bundle for driver $DRIVER_VERSION"
    echo "Expected: $NVIDIA_LIB_DIR"
    exit 2
fi

WORK_TMP="${SLURM_TMPDIR:-/tmp}/carla_runtime_${SLURM_JOB_ID:-$$}"
mkdir -p "$WORK_TMP/bin" "$WORK_TMP/runtime" "$RUN_DIR/home"

cat > "$WORK_TMP/bin/xdg-user-dir" <<'EOF'
#!/bin/sh
case "$1" in
  DESKTOP) echo "$HOME/Desktop" ;;
  DOWNLOAD) echo "$HOME/Downloads" ;;
  DOCUMENTS) echo "$HOME/Documents" ;;
  MUSIC) echo "$HOME/Music" ;;
  PICTURES) echo "$HOME/Pictures" ;;
  VIDEOS) echo "$HOME/Videos" ;;
  *) echo "$HOME" ;;
esac
EOF
chmod +x "$WORK_TMP/bin/xdg-user-dir"
chmod 700 "$WORK_TMP/runtime"

HOSTNAME_FQDN="$(hostname -f 2>/dev/null || hostname)"

cat > "$RUN_DIR/connection.env" <<EOF
CARLA_JOB_ID=${SLURM_JOB_ID:-manual}
CARLA_HOST=$HOSTNAME_FQDN
CARLA_PORT=$RPC_PORT
CARLA_LOG_DIR=$RUN_DIR
CARLA_MAP=$CARLA_MAP
EOF

echo "=== SimLingo CARLA runtime job ==="
echo "Host: $HOSTNAME_FQDN"
echo "Port: $RPC_PORT"
echo "Driver: $DRIVER_VERSION"
echo "Logs: $RUN_DIR"
echo "Connection file: $RUN_DIR/connection.env"
echo "CARLA SIF: $CARLA_SIF"
echo "NVIDIA libs: $NVIDIA_LIB_DIR"
if [ -n "$CARLA_MAP" ]; then
    echo "Startup map: $CARLA_MAP"
fi

# Launch the default world first. The requested map is loaded through the
# Python API after the server becomes ready because command-line map startup
# is unreliable with the packaged CARLA binary on this cluster.
CARLA_MAP_ARG=""

"$CONTAINER_CMD" exec --nv \
    --cleanenv \
    --env "XDG_RUNTIME_DIR=$WORK_TMP/runtime" \
    --bind "$CARLA_DIR:/opt/CARLA_0.9.15" \
    --bind "$NVIDIA_LIB_DIR:/project_nvidia_libs:ro" \
    --bind "$RUN_DIR:/run_output" \
    "$CARLA_SIF" \
    bash -lc "export HOME=/run_output/home; mkdir -p \$HOME; export PATH='$WORK_TMP/bin':/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; export LD_LIBRARY_PATH=/project_nvidia_libs:/.singularity.d/libs:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu; export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; cd /opt/CARLA_0.9.15; ./CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 $CARLA_MAP_ARG -RenderOffScreen -unattended -stdout -FullStdOutLogOutput -NoSplash -NoVSync -nosound -quality-level=Low -carla-rpc-port=$RPC_PORT -abslog=/run_output/CarlaUE4_runtime.log" \
    > "$RUN_DIR/carla_stdout.log" \
    2> "$RUN_DIR/carla_stderr.log" &
CARLA_PID=$!

echo "CARLA process PID: $CARLA_PID"
echo "Waiting for CARLA Python API readiness..."

eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$CARLA_PYTHON_CONDA_ENV"
ACTIVE_PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

mkdir -p "$CARLA_CLIENT_COMPAT_PY"
if [ -d "$CARLA_CLIENT_COMPAT_PY/carla" ] && [ -d "$CARLA_CLIENT_COMPAT_PY/carla.libs" ]; then
    CARLA_CLIENT_PYTHONPATH="$CARLA_CLIENT_COMPAT_PY"
elif [ -d "$CARLA_CLIENT_SOURCE_SITE/carla" ] && [ -d "$CARLA_CLIENT_SOURCE_SITE/carla.libs" ]; then
    ln -sfn "$CARLA_CLIENT_SOURCE_SITE/carla" "$CARLA_CLIENT_COMPAT_PY/carla"
    ln -sfn "$CARLA_CLIENT_SOURCE_SITE/carla.libs" "$CARLA_CLIENT_COMPAT_PY/carla.libs"
    CARLA_CLIENT_PYTHONPATH="$CARLA_CLIENT_COMPAT_PY"
elif [ "$ACTIVE_PYTHON_VERSION" = "3.7" ] && [ -e "$CARLA_PYTHON_EGG" ]; then
    CARLA_CLIENT_PYTHONPATH="$CARLA_ROOT/PythonAPI:$CARLA_PYTHON_EGG"
else
    echo "ERROR: no CARLA Python client package found."
    echo "Missing compatible package source: $CARLA_CLIENT_SOURCE_SITE/carla"
    echo "The fallback egg is Python 3.7-only and cannot be used from Python $ACTIVE_PYTHON_VERSION."
    exit 2
fi
export PYTHONPATH="$CARLA_CLIENT_PYTHONPATH:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:$CARLA_CLIENT_COMPAT_LIB:${LD_LIBRARY_PATH:-}"

READY=0
for attempt in $(seq 1 60); do
    if ! kill -0 "$CARLA_PID" 2>/dev/null; then
        wait "$CARLA_PID" || true
        echo "ERROR: CARLA exited before becoming ready."
        tail -160 "$RUN_DIR/carla_stderr.log" || true
        tail -160 "$RUN_DIR/carla_stdout.log" || true
        exit 1
    fi

    if python - "$RPC_PORT" <<'PY' >/dev/null 2>&1
import sys
import carla

port = int(sys.argv[1])
client = carla.Client("localhost", port)
client.set_timeout(2.0)
client.get_world()
PY
    then
        READY=1
        break
    fi
    sleep 2
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: CARLA did not become ready in time."
    tail -160 "$RUN_DIR/carla_stderr.log" || true
    tail -160 "$RUN_DIR/carla_stdout.log" || true
    exit 1
fi

if [ -n "$CARLA_MAP" ] && [ "$CARLA_MAP" != "current" ] && [ "$CARLA_MAP" != "default" ]; then
    echo "=== Loading requested map through Python API: $CARLA_MAP ==="
    if python - "$RPC_PORT" "$CARLA_MAP" <<'PY'
import sys
import time
import carla

port = int(sys.argv[1])
map_name = sys.argv[2]
client = carla.Client("localhost", port)
client.set_timeout(120.0)
world = client.load_world(map_name)
time.sleep(5.0)
world = client.get_world()
print(f"Loaded requested map: {world.get_map().name}", flush=True)
PY
    then
        echo "Requested map load complete."
    else
        echo "ERROR: requested map load failed: $CARLA_MAP"
        tail -160 "$RUN_DIR/carla_stderr.log" || true
        tail -160 "$RUN_DIR/carla_stdout.log" || true
        exit 1
    fi
fi

echo "=== READY ==="
echo "CARLA server is running at $HOSTNAME_FQDN:$RPC_PORT"
echo "Connection info:"
cat "$RUN_DIR/connection.env"
echo ""
if python - "$RPC_PORT" <<'PY'
import sys
import carla

port = int(sys.argv[1])
client = carla.Client("localhost", port)
client.set_timeout(5.0)
world = client.get_world()
print(f"Current map: {world.get_map().name}")
PY
then
    true
else
    echo "WARNING: could not query current map after readiness."
fi

echo "This job will keep CARLA alive until the Slurm time limit or cancellation."
echo "Cancel with: scancel ${SLURM_JOB_ID:-<jobid>}"

set +e
wait "$CARLA_PID"
CARLA_STATUS=$?
set -e

echo "=== CARLA process exited with status $CARLA_STATUS ==="
if [ "$CARLA_STATUS" -ne 0 ]; then
    echo "=== Tail: carla_stderr.log ==="
    tail -200 "$RUN_DIR/carla_stderr.log" || true
    echo "=== Tail: carla_stdout.log ==="
    tail -200 "$RUN_DIR/carla_stdout.log" || true
    echo "=== Tail: CarlaUE4_runtime.log ==="
    tail -200 "$RUN_DIR/CarlaUE4_runtime.log" || true
fi
exit "$CARLA_STATUS"
