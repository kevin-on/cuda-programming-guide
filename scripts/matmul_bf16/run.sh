#!/bin/bash
set -euo pipefail

BIN="build/matmul_bf16"
NVCC_FLAGS=(-O2 -std=c++17 -arch=sm_80 -lineinfo)
RUN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
    --debug)
        NVCC_FLAGS+=(-DDEBUG)
        shift
        ;;
    --nvcc-flag)
        shift
        [[ $# -gt 0 ]] || {
            echo "Missing value for --nvcc-flag" >&2
            exit 1
        }
        while [[ $# -gt 0 ]]; do
            case "$1" in
            -- | --debug | --nvcc-flag)
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
        RUN_ARGS+=("$@")
        break
        ;;
    *)
        RUN_ARGS+=("$1")
        shift
        ;;
    esac
done

mkdir -p build

nvcc "${NVCC_FLAGS[@]}" src/matmul_bf16/*.cu -o "$BIN" -lcublas
"$BIN" "${RUN_ARGS[@]}"
