#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SMARTSIM_DIR="${ROOT_DIR}/components/smartsim"
SMARTREDIS_DIR="${ROOT_DIR}/components/smartredis"
REDISAI_DIR="${ROOT_DIR}/components/redisai"

PYTHON="${PYTHON:-python3}"
SMART="${SMART:-$(dirname "${PYTHON}")/smart}"

export SMARTSIM_REDISAI_URL="${REDISAI_DIR}"

echo "SmartSim-CSC root: ${ROOT_DIR}"
echo "Python:           $(${PYTHON} --version 2>&1)"
echo

for directory in \
    "${SMARTSIM_DIR}" \
    "${SMARTREDIS_DIR}" \
    "${REDISAI_DIR}"
do
    if [[ ! -d "${directory}" ]]; then
        echo "Missing component directory: ${directory}" >&2
        exit 1
    fi
done

echo "Installing bundled SmartRedis and SmartSim..."
"${PYTHON}" -m pip install \
    --force-reinstall \
    "${SMARTREDIS_DIR}" \
    "${SMARTSIM_DIR}"

echo
echo "Installing JAX runtime..."
"${PYTHON}" -m pip install \
    jax \
    jaxlib

if [[ ! -x "${SMART}" ]]; then
    echo "SmartSim CLI was not found: ${SMART}" >&2
    exit 1
fi

echo
echo "Building Redis, RedisAI, ONNX Runtime, and JAX backends..."
"${SMART}" build \
    --device cpu \
    --skip-torch \
    --skip-tensorflow

echo
echo "Checking installed Python packages..."
"${PYTHON}" -m pip check

"${PYTHON}" - <<'PY'
from importlib.metadata import version
from pathlib import Path

from smartsim._core.config import CONFIG
from smartsim._core._install.buildenv import Versioner

for package in ("numpy", "jax", "jaxlib", "smartredis", "smartsim"):
    print(f"  {package}: {version(package)}")

redisai_source = Path(Versioner.REDISAI_URL).resolve()
required_paths = (
    Path(CONFIG.database_exe),
    Path(CONFIG.lib_path) / "redisai.so",
    Path(CONFIG.lib_path) / "backends" / "redisai_onnxruntime",
    Path(CONFIG.lib_path) / "backends" / "redisai_jax",
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
echo "Validating SmartSim runtime..."
"${SMART}" validate

echo
echo "SmartSim-CSC installation completed successfully."
