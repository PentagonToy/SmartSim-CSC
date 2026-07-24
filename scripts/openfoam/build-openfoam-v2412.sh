#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPONENT_DIR="$ROOT_DIR/components/openfoam-smartsim"

if [[ "${WM_PROJECT_VERSION:-}" != "v2412" ]]; then
    echo "OpenFOAM v2412 is not loaded." >&2
    exit 1
fi

if [[ -z "${SMARTREDIS_DIR:-}" ]]; then
    echo "SMARTREDIS_DIR is not defined." >&2
    exit 1
fi

export SMARTREDIS_INCLUDE="$SMARTREDIS_DIR/install/include"
export SMARTREDIS_LIB="$SMARTREDIS_DIR/install/lib64"

if [[ -z "${FOAM_USER_DIR:-}" ]]; then
    echo "FOAM_USER_DIR is not defined." >&2
    exit 1
fi

export FOAM_USER_APPBIN="$FOAM_USER_DIR/platforms/$WM_OPTIONS/bin"
export FOAM_USER_LIBBIN="$FOAM_USER_DIR/platforms/$WM_OPTIONS/lib"

mkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN"

OPENFOAM_SMARTSIM_LDFLAGS=""

IFS=':' read -ra LIB_DIRS <<< "${LD_LIBRARY_PATH:-}"
for dir in "${LIB_DIRS[@]}" /lib64; do
    [[ -d "$dir" ]] || continue
    OPENFOAM_SMARTSIM_LDFLAGS+=" -Wl,-rpath-link,$dir"
done

export OPENFOAM_SMARTSIM_LDFLAGS

cd "$COMPONENT_DIR/applications/utilities/foamSmartSimSvd"

wclean
wmake

echo
echo "Built:"
echo "$FOAM_USER_APPBIN/foamSmartSimSvd"
