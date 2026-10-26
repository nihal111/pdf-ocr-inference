"""Save page OCR candidates for visual review, never benchmark references or latency results."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback


def candidate(worker, pipeline, pdf, page, penalty, cap, output):
    os.environ["OMP_NUM_THREADS"] = "1"
    from document_pipeline.config import load_config
    from document_pipeline.ocr import HfOcr
    from document_pipeline.page_parallel import _parent_lifetime
    import pypdfium2 as pdfium
    import torch

    _parent_lifetime(os.getppid())
    torch.set_num_threads(1)
    directory = Path(output) / f"page-{page}-penalty-{penalty}"
    directory.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "page": page, "repetition_penalty": penalty,
              "device": worker, "max_new_tokens": cap, "purpose": "coverage review only"}
    (directory / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    try:
        cfg = load_config(pipeline)
        cfg.ocr.repetition_penalty = penalty
        cfg.ocr.max_new_tokens = cap
        # Save rejected candidates too, so a cap or repetition failure is inspectable.
        # This diagnostic setting is never used to capture baseline references.
        cfg.ocr.require_eos = False
        ocr = HfOcr(cfg.ocr, device=f"cuda:{worker}")
        with pdfium.PdfDocument(pdf) as doc:
            image = doc[page - 1].render(scale=cfg.render.dpi / 72).to_pil().convert("RGB")
        started = time.monotonic()
        text = ocr.ocr([image])[0]
        (directory / "page.md").write_text(text + "\n")
        status.update(status="awaiting review", stats=ocr.stats,
                      diagnostic_elapsed_s=time.monotonic() - started)
    except Exception:
        status.update(status="failed", error=traceback.format_exc())
    (directory / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pages", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--penalties", nargs="+", type=float, default=[1.05, 1.1])
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()
    exp = yaml.safe_load(Path(args.experiment).read_text())
    for key, value in exp["base"]["env"].items():
        os.environ[key] = str(value)
    tasks = [(page, penalty) for penalty in args.penalties for page in args.pages]
    if len(tasks) > 4:
        parser.error("At most four candidates: one model and one page on each GPU.")
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "probe.json").write_text(json.dumps(
        {"pipeline": exp["base"]["pipeline"], "tasks": tasks,
         "max_new_tokens": args.max_new_tokens, "is_baseline": False}, indent=2) + "\n")
    context = mp.get_context("spawn")
    processes = []
    try:
        for worker, (page, penalty) in enumerate(tasks):
            process = context.Process(target=candidate, args=(
                worker, exp["base"]["pipeline"], args.pdf, page, penalty,
                args.max_new_tokens, str(directory)))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
    return int(any(process.exitcode != 0 for process in processes))


if __name__ == "__main__":
    raise SystemExit(main())
