#!/bin/bash
set -euo pipefail

NVCC_FLAGS=(-g -lineinfo -cubin -arch=sm_80)
SRC=""
NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nvcc-flag)
      shift
      [[ $# -gt 0 ]] || {
        echo "Missing value for --nvcc-flag" >&2
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
      ;;
    -*)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 <source.cu> [output-name] [--nvcc-flag <flag> [flag...]]" >&2
      exit 1
      ;;
    *)
      if [[ -z "$SRC" ]]; then
        SRC="$1"
      elif [[ -z "$NAME" ]]; then
        NAME="$1"
      else
        echo "Unexpected argument: $1" >&2
        echo "Usage: $0 <source.cu> [output-name] [--nvcc-flag <flag> [flag...]]" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

[[ -n "$SRC" ]] || {
  echo "Usage: $0 <source.cu> [output-name] [--nvcc-flag <flag> [flag...]]" >&2
  exit 1
}
NAME="${NAME:-$(basename "${SRC}" .cu)}"

if [[ ! -f "$SRC" ]]; then
  echo "Error: $SRC not found"
  exit 1
fi

mkdir -p build logs

nvcc "${NVCC_FLAGS[@]}" "$SRC" -o "build/${NAME}.cubin"
nvdisasm -c -hex -gi "build/${NAME}.cubin" > "logs/${NAME}.cubin.log"

echo "Output: logs/${NAME}.cubin.log"
