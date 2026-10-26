"""Rerun the frozen pipeline on one PDF, retaining OCR text before EOS validation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.experiment).read_text())
    directory = Path(args.out).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for key, value in spec["base"]["env"].items():
        os.environ[key] = str(value)
    os.environ["DOCUMENT_PIPELINE_OCR_DIAGNOSTICS"] = str(directory / "pages")
    from document_pipeline.config import load_config
    from document_pipeline.pipeline import Pipeline
    from lab.pdf_env import collect
    cfg = load_config(spec["base"]["pipeline"])
    assert cfg.ocr.max_new_tokens == 8192 and cfg.ocr.repetition_penalty == 1.0
    assert cfg.ocr.require_eos and cfg.ocr.gpu_workers == 4
    record = {"status": "running", "pdf": str(Path(args.pdf).resolve()),
              "config": cfg.to_dict(), "environment": collect(cfg.to_dict(), os.environ),
              "purpose": "Inspect the latest 8192-token failure; not a baseline measurement"}
    status = directory / "diagnostic.json"
    status.write_text(json.dumps(record, indent=2) + "\n")
    pipe = None
    started = time.monotonic()
    try:
        pipe = Pipeline(cfg)
        pipe(args.pdf)
        record.update(status="finished", ocr_stats=pipe.ocr.stats)
    except Exception:
        record.update(status="rejected", error=traceback.format_exc())
    finally:
        if pipe is not None:
            pipe.ocr.close()
    record["elapsed_s_including_startup"] = time.monotonic() - started
    status.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in {"config", "environment"}}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
