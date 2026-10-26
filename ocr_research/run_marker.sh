#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$ROOT/.tools/llama-cuda/llama-b11838"
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$LLAMA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LLAMA_CPP_BINARY="$LLAMA_DIR/llama-server"
export LLAMA_CPP_NGL=99
export SURYA_INFERENCE_BACKEND=llamacpp
export SURYA_INFERENCE_PARALLEL=1
export SURYA_INFERENCE_KEEP_ALIVE=1
export LLAMA_CPP_EXTRA_ARGS="--image-min-tokens 1024"
export FAST_DETECTOR_DEVICE=cpu
export HF_HOME="$ROOT/.cache/hf"
cd "$ROOT"
for page in general thermo ir colloid; do
  echo "START $page $(date -Is)"
  ./.venv-marker/bin/marker_single "sample/pages/${page}.pdf" \
    --force_ocr \
    --mode fast \
    --disable_image_extraction \
    --output_format markdown \
    --output_dir "results/marker/${page}"
  echo "DONE $page $(date -Is)"
done
echo MARKER_ALL_DONE
