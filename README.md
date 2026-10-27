# inference-lab

## Static website

```bash
python3 -m lab.site --port 8766
```

Open `http://127.0.0.1:8766`. The committed `site/` directory contains the entire
website: HTML, CSS, JavaScript, JSON snapshots, PDF files and preview images. Any
static HTTP server can serve it; there is no application backend or inference at
runtime. The optional Python preview uses only the standard library and binds
to loopback.

The site includes a pipeline walkthrough, the baseline/evaluator contract, source-linked OCR
model options, a profiling guide, an experiment journal, and views of saved PDF sweep reports
and profiles. Hash routes and relative asset paths work both at a domain root and
under a GitHub Pages repository prefix.

Every served file must be **at most 1,000,000 bytes**. The site keeps aggregate
profiling results and the reports used by its views. Raw event traces, large
diagnostic logs, model files and benchmark reference vectors remain local and
are excluded from the export. The per-file limit is checked during export and CI.

### Update the snapshot

On the machine containing `results/` and `data/`:

```bash
python3 -m venv .venv-site
.venv-site/bin/pip install -r requirements-site.txt
.venv-site/bin/python -m lab.export_site
python3 -m lab.export_site --check site
```

Edit `web/` or `docs/`, re-export, then commit the updated `site/` alongside the
source changes. Exporting reads existing measurements only; it does not run
models or use GPUs. A clean checkout can preview, validate and deploy the
committed snapshot without the ignored local experiment files.

### GitHub Pages

In the GitHub repository, select **Settings → Pages → Source → GitHub Actions**.
The included workflow validates the snapshot and deploys `site/` on pushes to
`main`; it can also be started manually. GitHub serves the exported files directly.
No Python installation or API server is needed by visitors.

For a remote GPU host, forward the local preview with
`ssh -N -L 8766:127.0.0.1:8766 YOUR_GPU_HOST`.

**Workflow: baseline → profile → choose a hypothesis → change → validate → measure → record.**
Profile the baseline before choosing an optimization. Experiment YAMLs are candidate ideas,
not a mandatory sequence. Reprioritize them using the observed bottleneck.

Create an experiment note with
`.venv/bin/python -m lab.note experiment-name --title "The question we are testing"`.
The journal template records evidence, changes, commands, correctness, measurements, and decisions.

## Serving lab

A profiling and optimization lab for LLM serving. The baseline is
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (~1.4k lines, vendored in `engine/`),
which already has continuous batching, a paged KV cache, prefix caching, chunked prefill
(first sequence only), CUDA graphs, torch.compile'd ops and tensor parallelism. The lab adds a
harness that lets you switch each optimization on or off one at a time, measure what it is worth,
and then try to beat the baseline and vLLM.

```
engine/nanovllm/     the engine you modify (UPSTREAM = pinned commit, lab-hooks.patch = our only changes)
lab/workloads.py     deterministic synthetic workloads (offline + Poisson/bursty online)
lab/engine.py        LabEngine (real, instrumented) and MockEngine (real scheduler, fake GPU)
lab/bench.py         offline/online drivers + metrics
lab/run.py           one config x N workloads -> JSONL       lab/sweep.py   YAML experiment -> subprocess per variant
lab/compare.py       median-of-repeats tables, Δ vs baseline lab/prof.py    torch.profiler window -> Perfetto trace
experiments/*.yaml   ready-made ablations (00-07)            scripts/       GPU setup, model download, nsys
EXPERIMENTS.md       optimization backlog + results log
```

## PDF pipeline (`document_pipeline/` + `eval/`)

The active OCR stage is **[PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)** through its PaddleOCR page parser and vLLM recognition, with [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) for layout and reading order. The vLLM batch-1 baseline and batch-4 experiments are defined in `experiments/pdf/08_vllm_five_pdf_eval.yaml`; the original PaddlePaddle baseline is preserved. The engine-related output differences were accepted, with separate frozen vLLM references for future optimizations.

| Stage | Current baseline |
| --- | --- |
| Render | pypdfium2, 150 DPI RGB |
| OCR | Four L4 workers, each holding the PaddleOCR recognizer and layout detector; one page per GPU and one region per recognition batch; batch invariance enabled |
| Completion | Every region must emit EOS within 8,192 tokens, before parser text cleanup |
| Page assembly | Native page Markdown, restored to original PDF page order |
| Chunk | Markdown blocks with a 512-token target and heading context |
| Embed | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), sentence-transformers, fp32, on GPU 0 |

The evaluator calls `embed_pdf(path)` from `pipelines.baseline` or `pipelines.optimized`, both thin wrappers over `document_pipeline.adapter`. OCR bytes and chunk structure must match the reference exactly, and embedding cosine must be at least 0.99. The current evaluation uses five frozen PDFs (20 pages) and **one warm-up plus five measured calls per PDF**, sequentially. The evaluator pools the 25 measured full-PDF calls for p50/p95 and reports sequential pages/minute. Accepted speedups require matching hardware, dataset, references and settings. Results are never cached between calls.

`embed_pdf` returns full-document `ocr_text` and a non-empty, ordered `chunks` list.
Each chunk has a unique string `id` and a non-empty plain Python list of finite
floats in `embedding`; all vectors have the same length. Missing files raise
`FileNotFoundError`, and malformed PDFs raise `ValueError`. These input errors
leave the pipeline ready for another call. See the [Python interface](docs/pages/pipeline.md#python-interface).

### Baseline vs optimized with the evaluator

`pipelines.baseline` and `pipelines.optimized` pin the two measured configurations
(`vllm_batch_1` and `server_8_draft4_control`) and expose the same `embed_pdf`.
`tests/test_pipelines.py` fails if either drifts from its experiment YAML. Each variant
needs its own recognizer launch; stop the previous servers before starting the next.

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="$(python3 -c 'import json; print(json.load(open("reports/baseline-env.json"))["gpu_uuids"])')"
export CUDA_MODULE_LOADING=EAGER VLLM_BATCH_INVARIANT=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True VLLM_USE_FLASHINFER_SAMPLER=0
python3 -m unittest discover -s eval/tests
test -z "$(git status --porcelain)" || { echo "Commit changes before measuring"; exit 1; }

# Stage all measurements outside tracked reports so both runs measure a clean tree.
RUN_DIR="results/revalidation/$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
# Read the pinned literals without importing either GPU pipeline.
for variant in baseline optimized; do
  python3 - "$variant" > "$RUN_DIR/$variant-config.json" <<'PY'
import ast, json, sys
from pathlib import Path
root = Path.cwd()
tree = ast.parse((root / "pipelines" / (sys.argv[1] + ".py")).read_text())
config = next(ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and any(
                  isinstance(t, ast.Name) and t.id == "PIPELINE" for t in n.targets))
config["ocr"]["paddle_python"] = str(root / ".venv-paddle/bin/python")
config["ocr"]["paddle_server_python"] = str(root / ".venv-vllm/bin/python")
print(json.dumps(config))
PY
done

# Baseline recognizers: four requests per GPU, no speculation, no decoder tiles.
python -m lab.paddle_servers --directory "$RUN_DIR/baseline-servers" --max-num-seqs 4 &
SERVER_PID=$!
until test -f "$RUN_DIR/baseline-servers/ready.json"; do
  kill -0 "$SERVER_PID" || { wait "$SERVER_PID"; exit 1; }
  sleep 2
done
python -m lab.pdf_env --config "$(cat "$RUN_DIR/baseline-config.json")" \
  --out "$RUN_DIR/baseline-env.json"
# References are shipped. Only if absent, capture from the baseline:
# python3 eval/capture_reference.py --module pipelines.baseline \
#   --manifest data/public-v3/manifest.json --root data/public-v3 --timeout 1800
python3 eval/evaluate.py --module pipelines.baseline \
  --manifest data/public-v3/manifest.json --root data/public-v3 \
  --warmup 1 --repeats 5 --timeout 1800 \
  --environment "$RUN_DIR/baseline-env.json" --out "$RUN_DIR/baseline.json"
kill "$SERVER_PID"
wait "$SERVER_PID" || true

# Optimized recognizers: eight requests per GPU, four GPU n-gram drafts, fixed 32-row tiles.
python -m lab.paddle_servers --directory "$RUN_DIR/optimized-servers" \
  --max-num-seqs 8 --ngram-tokens 4 --ngram-method ngram_gpu --decode-matmul-fixed32 \
  --isolate-compile-cache &
SERVER_PID=$!
until test -f "$RUN_DIR/optimized-servers/ready.json"; do
  kill -0 "$SERVER_PID" || { wait "$SERVER_PID"; exit 1; }
  sleep 2
done
python -m lab.pdf_env --config "$(cat "$RUN_DIR/optimized-config.json")" \
  --out "$RUN_DIR/optimized-env.json"
python3 eval/evaluate.py --module pipelines.optimized \
  --manifest data/public-v3/manifest.json --root data/public-v3 \
  --warmup 1 --repeats 5 --timeout 1800 \
  --environment "$RUN_DIR/optimized-env.json" --out "$RUN_DIR/optimized.json"
kill "$SERVER_PID"
wait "$SERVER_PID" || true

python3 eval/compare.py --baseline "$RUN_DIR/baseline.json" \
  --optimized "$RUN_DIR/optimized.json" --out "$RUN_DIR/comparison.json"
# After reviewing the results, replace the reports and update their provenance.
cp "$RUN_DIR"/{baseline,optimized,comparison,baseline-env,optimized-env}.json reports/
```

Both server launches require the exported environment above; importing a client
module cannot change an already-running server's environment. The optimized
launch also preserves the original isolated compilation-cache policy. TTFR is
the evaluator's process-to-first-result time, including client model loading and
the first warm-up, but excluding recognizer-server startup.

Paddle layout/parser runs in `.venv-paddle`, recognizer servers in `.venv-vllm`, and the parent/embedder in `.venv`. Keep these environments separate because their CUDA-library packages share filenames.

```bash
uv venv --python 3.12 .venv-paddle
uv pip install --python .venv-paddle/bin/python 'paddleocr[doc-parser]==3.7.0' \
  'paddlepaddle-gpu @ https://paddle-whl.cdn.bcebos.com/stable/cu126/paddlepaddle-gpu/paddlepaddle_gpu-3.2.1-cp312-cp312-linux_x86_64.whl'
scripts/download_models.sh paddleocr layout embed

# Current frozen five-PDF evaluation (requires the four local vLLM recognizers):
.venv/bin/python -m lab.pdf_sweep experiments/pdf/08_vllm_five_pdf_eval.yaml
```

The selected PDFs and their references are frozen under `data/public-v3/`. The three newly added documents completed reference capture with 295 regions ending in EOS. A shared table-image limit of 2,822,400 pixels resolves the W-9 crop’s repeated-dot failure at the original limit. Reference review records ordinary OCR errors and chart/image placeholders; these are not character-perfect ground truth. See the [five-PDF evaluation journal](docs/experiments/009-five-pdf-evaluation.md) and [Paddle table-repetition investigation](docs/pages/debugging.md).

For the updated vLLM environment, model extension and baseline transition, see [the vLLM baseline journal](docs/experiments/007-vllm-batch4-baseline.md) and [engine setup](docs/experiments/006-vllm-region-batching.md). Four localhost recognizers must be running before using its experiment configuration.

See the [pipeline handbook](docs/pages/pipeline.md), [baseline contract](docs/pages/baseline.md), [PaddleOCR journal](docs/experiments/004-paddleocr-baseline.md), and [official PaddleOCR usage guide](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html). Profile the accepted baseline before choosing optimization experiments.

## Laptop (no GPU)

The mock engine runs nano-vllm's real `Scheduler`/`BlockManager` with a virtual clock and a toy
cost model. Use it to develop harness and scheduler changes. **Its numbers don't tell you
anything about GPU performance.**

```bash
uv run --extra dev --extra pdf pytest                     # ~3s; DOCUMENT_PIPELINE_SLOW=1 adds a real-pipeline test
uv run python -m lab.sweep experiments/01_knobs.yaml --mock --repeats 1
```

## L4 box

```bash
scripts/setup_gpu.sh                      # .venv (nano-vllm) + .venv-vllm (reference); SKIP_VLLM=1 to skip
scripts/download_models.sh 0.6B 8B embed paddleocr layout   # text models, embedder, PaddleOCR and layout detector
source .venv/bin/activate

python -m lab.sweep experiments/00_smoke.yaml            # sanity check
python -m lab.sweep experiments/01_knobs.yaml            # what each built-in optimization is worth
.venv-vllm/bin/python -m lab.sweep experiments/07_vllm_reference.yaml
python -m lab.compare results/01_knobs results/07_vllm_reference
```

Each sweep writes `results/<name>/<timestamp>/`: `results.jsonl` (one row per run, with config,
env, GPU and versions), `steps/*.json` (per-step trace for plotting batch size and step time over
time), per-variant logs, a copy of the YAML, and `git diff` of `engine/` if the repo is under git.
Pin a GPU with `CUDA_VISIBLE_DEVICES=1 python -m lab.sweep ...`, so you can run two sweeps on two L4s
in parallel. Don't run TP sweeps in parallel: nano-vllm hard-codes port 2333.

## The loop

1. **Baseline**: freeze the dataset/configuration, run the baseline, and preserve its reports.
   For the PDF pipeline use `experiments/pdf/00_baseline.yaml`; for the separate serving lab
   start with the baseline variant only.
2. **Profile before optimization**: find where time goes (see below) and write down a hypothesis: *"decode at bs=256 is
   CPU-bound in `prepare_decode`; making block tables persistent will cut step time X%"*.
3. **Change** `engine/nanovllm/...`, preferably behind a `Config` flag so it can be ablated like
   the built-in knobs.
4. **Measure**: add a variant to an experiment YAML (or make a new one) and compare. PDF official
   comparisons require at least one warm-up and five measured repeats. The serving lab uses
   its own harness. Inspect repeat variability before treating a small change as an improvement.
5. **Log** the result in `docs/experiments/`, including what did not work. Choose the next
   experiment from the evidence; the backlog is a reference blueprint.

## Metrics (lab/bench.py)

| metric | meaning |
|---|---|
| `output_tok_s`, `total_tok_s`, `req_s` | throughput over the measured run (warmup excluded) |
| `ttft_s`, `tpot_s`, `itl_s`, `e2e_s`, `queue_s` | p50/p90/p99/mean. TTFT is measured from *scheduled* arrival, so queueing counts |
| `slo_attainment`, `goodput_req_s` | share and rate of requests meeting the workload's TTFT+TPOT SLO |
| `mean_decode_batch` | average sequences per decode step: the main throughput lever |
| `decode_bw_gbps` | (non-embedding weights + KV read) / decode step time. Compare with ~300 GB/s on an L4 |
| `cpu_sched_post_frac` | share of busy time spent in Python scheduling and postprocessing |
| `prefill_compute_ratio` | prefill tokens computed / prompt tokens. <1 means prefix-cache hits, >1 means preemption recompute |
| `preemptions`, `peak_mem_gb`, `idle_time_s` | — |
| `breakdown_s` | with `flags: [--breakdown]`: sync-timed prepare / forward / sample |

Back-of-envelope ceilings for decode on an L4 (≈300 GB/s): Qwen3-0.6B streams about 1.2 GB per
step, a ~4 ms floor or ~250 tok/s at batch 1. Qwen3-8B streams about 16 GB, a ~55 ms floor or
~18 tok/s at batch 1. Batching amortizes that floor until you become compute-bound or run out
of KV cache.

## Profiling

For the PDF pipeline, `lab.pdf_prof` preserves each session in
`results/profiles/<session>/`: `profile.json`, `stages.jsonl`, `trace.json`, and `operators.json`.
The profile contains configuration, environment, PDF hash, and reproduction command. The
stage measurements exclude warm-up and the extra profiled call. Profiles are diagnostic;
use the evaluator for official speedups.

```bash
CUDA_VISIBLE_DEVICES=0 python -m lab.pdf_prof data/real/pdfs/example.pdf \
  --label baseline --baseline-run pdf_00_baseline/RUN_TIMESTAMP
```

For the separate serving lab:

```bash
# torch.profiler over 20 decode steps, once the batch is full -> results/traces/*.json (ui.perfetto.dev)
python -m lab.prof --workload decode_heavy --skip 300 --steps 20 --engine-args '{"enforce_eager": true}'
python -m lab.prof --workload prefill_heavy --skip 0 --steps 5

# Nsight Systems: CPU launch gaps, H2D copies, NCCL (needs nsys on PATH)
scripts/nsys.sh decode --workloads '{"use":"decode_heavy","num_requests":64}'

# Nsight Compute on one kernel, e.g. the KV-store Triton kernel
ncu --kernel-name regex:store_kvcache -c 5 python -m lab.run --workloads smoke
```

With CUDA graphs on, a decode step is a single graph launch, so profile with
`enforce_eager: true` to see individual kernels, then confirm any gain with graphs on.

## Caveats

- nano-vllm supports Qwen3 dense models only (0.6B–32B). Adding a Llama-style model means
  writing a new file in `engine/nanovllm/models/` and is a reasonable exercise in itself.
- Greedy sampling isn't allowed (temperature > 0). All workloads set `ignore_eos`, so output
  lengths are fixed.
- Prompts are random token ids, which is fine for throughput and latency but useless for
  speculative decoding acceptance rates. Use real text for that.
- `lab/engine.py` re-implements `LLMEngine.step` to add timing. If you change `step()` upstream,
  mirror the change in `Recorder.step`.
