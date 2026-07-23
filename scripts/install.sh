#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SMARTSIM_DIR="${ROOT_DIR}/components/smartsim"
SMARTREDIS_DIR="${ROOT_DIR}/components/smartredis"
REDISAI_DIR="${ROOT_DIR}/components/redisai"

PYTHON="${PYTHON:-python3}"

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
echo "Installed packages:"
"${PYTHON}" - <<'PY'
from importlib.metadata import version

for package in ("smartredis", "smartsim"):
    print(f"  {package}: {version(package)}")
PY

echo
echo "Bundled RedisAI source:"
"${PYTHON}" - <<'PY'
from pathlib import Path

from smartsim._core._install.buildenv import Versioner

source = Path(Versioner.REDISAI_URL).resolve()

print(f"  {source}")

if not source.is_dir():
    raise SystemExit(f"Bundled RedisAI source does not exist: {source}")
PY
