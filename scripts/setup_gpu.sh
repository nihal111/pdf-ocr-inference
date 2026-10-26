#!/usr/bin/env bash
# One-time setup on the L4 box. Creates .venv (nano-vllm + lab) and .venv-vllm (reference).
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv .venv --python 3.12
uv pip install --python .venv numpy xxhash pyyaml pytest "torch>=2.4" "triton>=3.0" "transformers>=4.51" safetensors tqdm \
  "sentence-transformers>=5" pypdfium2 pillow torchvision
# flash-attn compiles against the installed torch; prebuilt wheels are fetched when one matches.
uv pip install --python .venv flash-attn --no-build-isolation

# vLLM pins its own torch, so it gets a separate env.
if [[ "${SKIP_VLLM:-0}" != "1" ]]; then
  uv venv .venv-vllm --python 3.12
  # The PDF pipeline's vllm OCR backend runs inside this env, so it needs the pipeline deps too.
  uv pip install --python .venv-vllm vllm pyyaml xxhash "sentence-transformers>=5" pypdfium2 pillow
fi

.venv/bin/python - <<'PY'
import torch, flash_attn
print("torch", torch.__version__, "cuda", torch.version.cuda, "flash_attn", flash_attn.__version__)
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
nvidia-smi topo -m || true   # shows how the L4s are connected (PIX/PHB/SYS = PCIe paths)
