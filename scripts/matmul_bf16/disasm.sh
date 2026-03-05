#!/bin/bash
set -euo pipefail

SRC="${1:?Usage: $0 <source.cu> [output-name]}"
NAME="${2:-$(basename "${SRC}" .cu)}"

if [[ ! -f "$SRC" ]]; then
  echo "Error: $SRC not found"
  exit 1
fi

mkdir -p build logs

nvcc -g -lineinfo -cubin -arch=sm_80 "$SRC" -o "build/${NAME}.cubin"
nvdisasm -c -hex -gi "build/${NAME}.cubin" > "logs/${NAME}.cubin.log"

echo "Output: logs/${NAME}.cubin.log"
