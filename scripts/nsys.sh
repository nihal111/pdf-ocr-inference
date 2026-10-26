#!/usr/bin/env bash
# Nsight Systems timeline of one measured run (CPU launch gaps, kernel overlap, NCCL).
# Usage: scripts/nsys.sh <name> [lab.run args...]
#   scripts/nsys.sh decode_eager --workloads '{"use":"decode_heavy","num_requests":64}' --engine-args '{"enforce_eager":true}'
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1; shift
mkdir -p results/traces
nsys profile -o "results/traces/$name" --force-overwrite true \
  --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi --capture-range-end=stop \
  .venv/bin/python -m lab.run --nvtx --cuda-profiler-range "$@"
nsys stats --report cuda_gpu_kern_sum,nvtx_sum "results/traces/$name.nsys-rep" | head -60
