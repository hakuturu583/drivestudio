#!/usr/bin/env bash
# Run DriveStudio training inside the Docker image.
# Usage:
#   scripts/train.sh --data /abs/path/to/data --project recon --run-name exp1 --scene-idx 0
# Options let you change config, dataset, timesteps, and CPU/GPU usage.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/train.sh [options]

Options:
  --data <path>        Absolute path to dataset root (mounted to /workspace/drivestudio/data) [default: REPO_ROOT/data]
  --output <path>      Absolute path for logs/checkpoints (mounted to /workspace/drivestudio/logs) [default: REPO_ROOT/logs]
  --config <file>      Config file to use (default: configs/omnire.yaml)
  --dataset <name>     Dataset config key, e.g. waymo/3cams (default: waymo/3cams)
  --scene-idx <n>      Scene index (default: 0)
  --start <n>          Start timestep (default: 0)
  --end <n>            End timestep, -1 for last frame (default: -1)
  --project <name>     Project name for logging (default: recon)
  --run-name <name>    Run/experiment name (default: exp)
  --image <tag>        Docker image to use (default: drivestudio:base)
  --cpu                Run without --gpus all
  --                   Pass all remaining args directly to tools/train.py
  -h, --help           Show this help

Example:
  scripts/train.sh --data /data/waymo --output /logs/drivestudio \\
    --project recon --run-name scene0 --scene-idx 0 --dataset waymo/3cams
USAGE
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_HOST="${REPO_ROOT}/data"
OUTPUT_HOST="${REPO_ROOT}/logs"
CONFIG_FILE="configs/omnire.yaml"
DATASET="waymo/3cams"
SCENE_IDX=0
START_TS=0
END_TS=-1
PROJECT="recon"
RUN_NAME="exp"
IMAGE="drivestudio:base"
GPU_FLAG="--gpus all"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA_HOST="$2"; shift 2;;
    --output) OUTPUT_HOST="$2"; shift 2;;
    --config) CONFIG_FILE="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --scene-idx) SCENE_IDX="$2"; shift 2;;
    --start) START_TS="$2"; shift 2;;
    --end) END_TS="$2"; shift 2;;
    --project) PROJECT="$2"; shift 2;;
    --run-name) RUN_NAME="$2"; shift 2;;
    --image) IMAGE="$2"; shift 2;;
    --cpu) GPU_FLAG=""; shift;;
    --) shift; EXTRA_ARGS=("$@"); break;;
    -h|--help) usage; exit 0;;
    *) echo "Unexpected argument: $1" >&2; usage; exit 1;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but not found in PATH." >&2
  exit 1
fi

DATA_REALPATH="$(python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "${DATA_HOST}")"
OUTPUT_REALPATH="$(python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "${OUTPUT_HOST}")"

mkdir -p "${OUTPUT_REALPATH}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Building Docker image ${IMAGE} ..."
  (cd "${REPO_ROOT}" && docker buildx bake base)
fi

# Ensure SMPL model is present and valid on host (download if missing/invalid)
SMPL_LOCAL="${REPO_ROOT}/smpl_models/SMPL_NEUTRAL.pkl"
validate_smpl() {
  local path="$1"
  python3 - <<PY "$path"
import os, pickle, sys
path = sys.argv[1]
if not os.path.isfile(path):
    sys.exit(1)
try:
    with open(path, "rb") as f:
        pickle.load(f, encoding="latin1")
    sys.exit(0)
except Exception as exc:
    sys.stderr.write(f"[warn] Invalid SMPL file {path}: {exc}\n")
    sys.exit(2)
PY
}

ensure_smpl() {
  local env_url="${SMPL_NEUTRAL_URL:-https://smpl.is.tue.mpg.de/download.php?filename=SMPL_python_v.1.1.0.zip}"
  local env_gid="${SMPL_NEUTRAL_GDRIVE_ID:-}"
  local strict="${SMPL_DOWNLOAD_STRICT:-1}"
  echo "Attempting to download SMPL_NEUTRAL.pkl using ${IMAGE} (URL=${env_url}, GDRIVE_ID=${env_gid}, STRICT=${strict})"
  mkdir -p "$(dirname "${SMPL_LOCAL}")"
  docker run --rm \
    -e SMPL_NEUTRAL_URL="${env_url}" \
    -e SMPL_NEUTRAL_GDRIVE_ID="${env_gid}" \
    -e SMPL_DOWNLOAD_STRICT="${strict}" \
    -v "${REPO_ROOT}:/workspace/drivestudio" \
    "${IMAGE}" bash /workspace/drivestudio/docker/fetch_smpl.sh
}

if ! validate_smpl "${SMPL_LOCAL}"; then
  echo "SMPL_NEUTRAL.pkl missing or invalid locally; downloading..."
  ensure_smpl || true
fi

if ! validate_smpl "${SMPL_LOCAL}"; then
  echo "SMPL_NEUTRAL.pkl is missing or invalid. Please place a valid file at ${SMPL_LOCAL} or set SMPL_NEUTRAL_URL/SMPL_NEUTRAL_GDRIVE_ID." >&2
  exit 1
fi

echo "Running training inside ${IMAGE} ..."
docker run ${GPU_FLAG} --rm -it \
  -v "${REPO_ROOT}:/workspace/drivestudio" \
  -v "${DATA_REALPATH}:/workspace/drivestudio/data" \
  -v "${OUTPUT_REALPATH}:/workspace/drivestudio/logs" \
  -e PYTHONPATH=/workspace/drivestudio \
  "${IMAGE}" bash -c "
    set -euo pipefail
    export PYTHONPATH=/workspace/drivestudio
    start_timestep=${START_TS}
    end_timestep=${END_TS}
    python3 tools/train.py \
      --config_file ${CONFIG_FILE} \
      --output_root /workspace/drivestudio/logs \
      --project ${PROJECT} \
      --run_name ${RUN_NAME} \
      dataset=${DATASET} \
      data.scene_idx=${SCENE_IDX} \
      data.start_timestep=\${start_timestep} \
      data.end_timestep=\${end_timestep} \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
  "
