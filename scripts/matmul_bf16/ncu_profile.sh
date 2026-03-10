#!/bin/bash
set -euo pipefail

BIN="build/matmul_bf16"
SRC_DIR="src/matmul_bf16"
NVCC_FLAGS=(-O2 -lineinfo -std=c++17 -arch=sm_80)
ARGS=()

usage() {
  echo "Usage: $0 [m] <kernel-spec> [label] [-- ncu-extra-args...] [--nvcc-flag <flag> [flag...]]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nvcc-flag)
      shift
      [[ $# -gt 0 ]] || {
        echo "Missing value for --nvcc-flag" >&2
        usage
        exit 1
      }
      while [[ $# -gt 0 ]]; do
        case "$1" in
          -- | --nvcc-flag)
            break
            ;;
          *)
            NVCC_FLAGS+=("$1")
            shift
            ;;
        esac
      done
      ;;
    --)
      shift
      ARGS+=("$@")
      break
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

set -- "${ARGS[@]}"

M=4096
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  M="$1"
  shift
fi

[[ $# -gt 0 ]] || {
  usage
  exit 1
}
SPEC="$1"
shift

LABEL=""
if [[ $# -gt 0 && "$1" != -- && "$1" != -* ]]; then
  LABEL="$1"
  shift
fi

[[ $# -gt 0 && "$1" == "--" ]] && shift
NCU_EXTRA_ARGS=("$@")

mkdir -p build

nvcc "${NVCC_FLAGS[@]}" "$SRC_DIR"/*.cu -o "$BIN" -lcublas

SUFFIX=$(echo "$SPEC" | tr ':=,' '_')
[[ -n "$LABEL" ]] && SUFFIX="${SUFFIX}_${LABEL}"
PROFILE_DIR="profiles/matmul_bf16/${SUFFIX}"
mkdir -p "$PROFILE_DIR"
REPORT="${PROFILE_DIR}/profile.ncu-rep"

cp "$SRC_DIR"/*.cu "$SRC_DIR"/*.cuh "$PROFILE_DIR/"

ncu --set full \
    --export "$REPORT" \
    -f \
    "${NCU_EXTRA_ARGS[@]}" \
    "$BIN" "$M" --warmup 0 --iters 1 --run "$SPEC"

echo "Report: $REPORT"
echo "Open with: ncu-ui $REPORT"
