#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-python3}"
PROFILE="${PROFILE:-linux-x64-cpu}"
SMART="${SMART:-$(dirname "${PYTHON}")/smart}"

while IFS='=' read -r key value
do
    case "${key}" in
        STACK_NAME) STACK_NAME="${value}" ;;
        STACK_VERSION) STACK_VERSION="${value}" ;;
        SMARTSIM_DIR) SMARTSIM_DIR="${value}" ;;
        SMARTREDIS_DIR) SMARTREDIS_DIR="${value}" ;;
        REDISAI_DIR) REDISAI_DIR="${value}" ;;
        DEVICE) DEVICE="${value}" ;;
        BACKENDS) BACKENDS="${value}" ;;
    esac
done < <(
    "${PYTHON}" "${ROOT_DIR}/scripts/stack_config.py" \
        --shell \
        --profile "${PROFILE}"
)

for variable in \
    STACK_NAME \
    STACK_VERSION \
    SMARTSIM_DIR \
    SMARTREDIS_DIR \
    REDISAI_DIR \
    DEVICE \
    BACKENDS
do
    if [[ -z "${!variable:-}" ]]; then
        echo "Missing stack configuration value: ${variable}" >&2
        exit 1
    fi
done

export SMARTSIM_REDISAI_URL="${REDISAI_DIR}"

echo "${STACK_NAME} ${STACK_VERSION}"
echo "Profile:          ${PROFILE}"
echo "Python:           $(${PYTHON} --version 2>&1)"
echo "Device:           ${DEVICE}"
echo "Backends:         ${BACKENDS}"
echo

echo "Installing bundled SmartRedis and SmartSim..."
"${PYTHON}" -m pip install \
    --force-reinstall \
    --no-deps \
    "${SMARTREDIS_DIR}" \
    "${SMARTSIM_DIR}"

IFS=',' read -ra BACKEND_LIST <<< "${BACKENDS}"

for backend in "${BACKEND_LIST[@]}"
do
    case "${backend}" in
        jax|onnxruntime|libtorch|libtensorflow) ;;
        *)
            echo "Unsupported backend in profile ${PROFILE}: ${backend}" >&2
            exit 1
            ;;
    esac
done

BUILD_ARGS=(
    build
    --device "${DEVICE}"
)

has_backend() {
    local target="$1"
    local backend

    for backend in "${BACKEND_LIST[@]}"
    do
        if [[ "${backend}" == "${target}" ]]; then
            return 0
        fi
    done

    return 1
}

if has_backend "jax"; then
    echo
    echo "Installing JAX runtime..."
    if [[ "${DEVICE}" == "cuda-12" ]]; then
        "${PYTHON}" -m pip install "jax[cuda12]"
    else
        "${PYTHON}" -m pip install jax jaxlib
    fi
else
    BUILD_ARGS+=(--skip-jax)
fi

if ! has_backend "onnxruntime"; then
    BUILD_ARGS+=(--skip-onnx)
fi

if ! has_backend "libtorch"; then
    BUILD_ARGS+=(--skip-torch)
fi

if ! has_backend "libtensorflow"; then
    BUILD_ARGS+=(--skip-tensorflow)
fi

if [[ ! -x "${SMART}" ]]; then
    echo "SmartSim CLI was not found: ${SMART}" >&2
    exit 1
fi

echo
echo "Building SmartSim dependencies..."
"${SMART}" "${BUILD_ARGS[@]}"

echo
echo "Checking installed Python packages..."
"${PYTHON}" -m pip check

"${PYTHON}" - <<'PY'
from importlib.metadata import version
from pathlib import Path

from smartsim._core.config import CONFIG
from smartsim._core._install.buildenv import Versioner

for package in ("numpy", "smartredis", "smartsim"):
    print(f"  {package}: {version(package)}")

redisai_source = Path(Versioner.REDISAI_URL).resolve()
required_paths = (
    Path(CONFIG.database_exe),
    Path(CONFIG.lib_path) / "redisai.so",
)

print(f"\nBundled RedisAI source:\n  {redisai_source}")

if not redisai_source.is_dir():
    raise SystemExit(f"Bundled RedisAI source does not exist: {redisai_source}")

for path in required_paths:
    print(f"  {path}: {path.exists()}")
    if not path.exists():
        raise SystemExit(f"Required build artifact does not exist: {path}")
PY

echo
if [[ "${DEVICE}" == "cuda-12" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo "Validating SmartSim runtime on GPU..."
        "${SMART}" validate --device gpu
    else
        echo "Skipping GPU runtime validation because no allocated GPU is visible."
        echo "Run the following command on a GPU node:"
        echo "  ${SMART} validate --device gpu"
    fi
else
    echo "Validating SmartSim runtime on CPU..."
    "${SMART}" validate --device cpu
fi

echo
echo "${STACK_NAME} ${STACK_VERSION} installation completed successfully."
