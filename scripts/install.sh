#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SMARTSIM_DIR="${ROOT_DIR}/components/smartsim"
SMARTREDIS_DIR="${ROOT_DIR}/components/smartredis"
REDISAI_DIR="${ROOT_DIR}/components/redisai"

PYTHON="${PYTHON:-python3}"

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
echo "Installed packages:"
"${PYTHON}" - <<'PY'
from importlib.metadata import version

for package in ("smartredis", "smartsim"):
    print(f"  {package}: {version(package)}")
PY

echo
echo "Bundled RedisAI source:"
echo "  ${REDISAI_DIR}"
