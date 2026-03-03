#!/bin/bash
set -euo pipefail

BIN="build/matmul"
M="${1:-4096}"
SPEC="${2:?Usage: $0 [m] <kernel-spec> [label] [ncu-extra-args...]}"
LABEL="${3:-}"
shift 2
[[ -n "$LABEL" ]] && shift

SRC_DIR="src/matmul_fp32"

mkdir -p build

nvcc -O2 -lineinfo -std=c++17 -arch=sm_80 "$SRC_DIR"/*.cu -o "$BIN" -lcublas

SUFFIX=$(echo "$SPEC" | tr ':=,' '_')
[[ -n "$LABEL" ]] && SUFFIX="${SUFFIX}_${LABEL}"
PROFILE_DIR="profiles/matmul_fp32/${SUFFIX}"
mkdir -p "$PROFILE_DIR"
REPORT="${PROFILE_DIR}/profile.ncu-rep"

cp "$SRC_DIR"/*.cu "$SRC_DIR"/*.cuh "$PROFILE_DIR/"

ncu --set full \
    --export "$REPORT" \
    -f \
    "$@" \
    "$BIN" "$M" --warmup 0 --iters 1 --run "$SPEC"

echo "Report: $REPORT"
echo "Open with: ncu-ui $REPORT"
