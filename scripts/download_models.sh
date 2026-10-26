#!/usr/bin/env bash
# Usage: scripts/download_models.sh [0.6B 1.7B 4B 8B embed paddleocr layout vl vl4b]
#   <size> -> Qwen/Qwen3-<size>               (LLM serving lab)
#   embed  -> Qwen/Qwen3-Embedding-0.6B       (PDF pipeline: embeddings)
#   paddleocr -> PaddlePaddle/PaddleOCR-VL-1.6 (PDF pipeline: OCR baseline)
#   layout -> PaddlePaddle/PP-DocLayoutV3      (PDF pipeline: layout detector)
#   vl     -> Qwen/Qwen3-VL-2B-Instruct       (historical OCR experiment)
#   vl4b   -> Qwen/Qwen3-VL-4B-Instruct
set -euo pipefail
names=("$@"); [[ ${#names[@]} -eq 0 ]] && names=(0.6B embed paddleocr layout)
for n in "${names[@]}"; do
  revision=()
  case $n in
    embed) repo=Qwen/Qwen3-Embedding-0.6B ;;
    paddleocr) repo=PaddlePaddle/PaddleOCR-VL-1.6; revision=(--revision c5630abae1d940eafe0697512a0325494b02ab42) ;;
    layout) repo=PaddlePaddle/PP-DocLayoutV3; revision=(--revision 241f8bdfc77a7c7bee915a5057aaee58c235a8d3) ;;
    vl)    repo=Qwen/Qwen3-VL-2B-Instruct ;;
    vl4b)  repo=Qwen/Qwen3-VL-4B-Instruct ;;
    *)     repo=Qwen/Qwen3-$n ;;
  esac
  dir=~/huggingface/${repo##*/}
  if command -v hf >/dev/null; then hf download "$repo" "${revision[@]}" --local-dir "$dir"
  else uvx --from huggingface_hub hf download "$repo" "${revision[@]}" --local-dir "$dir"; fi
done
