"""document_pipeline config + the eval-facing lab tools. Fast by default; DOCUMENT_PIPELINE_SLOW=1 also runs the real
pipeline (Qwen3-VL-2B + Qwen3-Embedding, models must be downloaded; slow on a laptop)."""
import json
import os
from pathlib import Path

import pytest

from lab.pdf_sweep import main as pdf_sweep, stage_summary
from document_pipeline.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_config_defaults_env_and_overrides(monkeypatch, tmp_path):
    monkeypatch.delenv("DOCUMENT_PIPELINE_CONFIG", raising=False)
    cfg = load_config()
    assert (cfg.embed.backend, cfg.embed.dtype, cfg.ocr.backend, cfg.ocr.page_batch_size) == \
        ("sentence_transformers", "float32", "paddleocr", 1)
    monkeypatch.setenv("DOCUMENT_PIPELINE_CONFIG", '{"embed": {"dtype": "bfloat16"}}')
    assert load_config().embed.dtype == "bfloat16"
    assert load_config({"embed": {"batch_size": 7}}).embed.batch_size == 7   # explicit beats env
    f = tmp_path / "c.yaml"
    f.write_text("ocr:\n  page_batch_size: 2\n")
    monkeypatch.setenv("DOCUMENT_PIPELINE_CONFIG", str(f))
    assert load_config().ocr.page_batch_size == 2
    with pytest.raises(ValueError, match="unknown EmbedConfig"):
        load_config({"embed": {"dtyp": "x"}})


@pytest.mark.parametrize("generated,finished", [([7, 8, 9], False), ([7, 99, 0], True)])
def test_hf_ocr_rejects_unfinished_pages(generated, finished, monkeypatch, tmp_path):
    import torch
    from types import SimpleNamespace
    from document_pipeline.ocr import HfOcr

    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        tokenizer = SimpleNamespace(pad_token_id=0)

        def __call__(self, **kwargs):
            return Inputs(input_ids=torch.tensor([[11, 12]]))

        def batch_decode(self, values, **kwargs):
            return ["transcribed text"]

    ocr = HfOcr.__new__(HfOcr)
    ocr.cfg = load_config({"ocr": {"max_new_tokens": 3, "repetition_penalty": 1.05}}).ocr
    ocr.device, ocr.torch, ocr.proc, ocr.text = "cpu", torch, Processor(), "prompt"
    def generate(**kwargs):
        assert kwargs["do_sample"] is False
        assert kwargs["repetition_penalty"] == 1.05
        return torch.tensor([[11, 12, *generated]])

    ocr.model = SimpleNamespace(config=SimpleNamespace(image_token_id=11),
                               generation_config=SimpleNamespace(eos_token_id=[99]),
                               generate=generate)
    monkeypatch.setenv("DOCUMENT_PIPELINE_OCR_DIAGNOSTICS", str(tmp_path))
    if finished:
        assert ocr.ocr(["image"]) == ["transcribed text"]
        assert ocr.stats["batches"][0]["ended_with_eos"] == [True]
    else:
        with pytest.raises(RuntimeError, match="Refusing truncated output"):
            ocr.ocr(["image"])
    saved = json.loads((tmp_path / "cpu/batch-1-page-1.json").read_text())
    assert saved["ended_with_eos"] == finished
    assert saved["generated_tokens"] == generated
    assert (tmp_path / "cpu/batch-1-page-1.md").read_text() == "transcribed text\n"


def test_stage_summary_skips_warmup(tmp_path):
    trace = tmp_path / "t.jsonl"
    rows = [("a.pdf", 10.0), ("a.pdf", 1.0), ("a.pdf", 3.0), ("b.pdf", 20.0), ("b.pdf", 2.0)]
    trace.write_text("".join(json.dumps({"pdf": p, "total_s": t, "stages_s": {"convert": t}}) + "\n" for p, t in rows))
    s = stage_summary(str(trace), warmup=1)
    assert s["total"] == pytest.approx(2.0) and s["convert"] == pytest.approx(2.0)


def test_capture_only_stops_before_evaluation(tmp_path, monkeypatch):
    from lab import pdf_sweep as module
    exp = tmp_path / "capture.yaml"
    exp.write_text("name: review\nvariants:\n  baseline: {}\n")
    called = []
    monkeypatch.setattr(module, "capture", lambda *args: called.append(args) or 0)
    monkeypatch.setattr(module, "_quiet_run", lambda *args: pytest.fail("Evaluation must not start"))
    assert module.main([str(exp), "--capture-only", "--out-root", str(tmp_path / "out")]) == 0
    run = next((tmp_path / "out/review").iterdir())
    assert len(called) == 1
    assert json.loads((run / "run-status.json").read_text())["status"] == "awaiting_review"
    assert not (run / "baseline.json").exists()
    with pytest.raises(SystemExit):
        module.main([str(exp), "--resume-run", str(run)])
    (run / "quality-review.json").write_text('{"status":"complete"}')
    exp.write_text("name: changed\nvariants:\n  baseline: {}\n")
    with pytest.raises(SystemExit):
        module.main([str(exp), "--resume-run", str(run)])


def test_pdf_env_fields():
    from lab.pdf_env import collect
    env = collect({"embed": {"dtype": "bfloat16"}})
    example = json.loads((ROOT / "eval/environment.example.json").read_text())
    assert set(example) <= set(env)
    assert env["concurrency"] == 1 and env["precision"] == "ocr=bfloat16 embed=bfloat16" and env["hardware_id"]


def test_pdf_sweep_with_smoke_adapter(tmp_path, capsys, monkeypatch):
    from lab import pdf_sweep as module
    exp = tmp_path / "exp.yaml"
    exp.write_text(f"""
name: t
module: eval.smoke_adapter
manifest: {ROOT}/eval/fixtures/manifest.json
root: {ROOT}/eval/fixtures
warmup: 1
repeats: 2
timeout: 60
variants:
  baseline: {{}}
  other: {{pipeline: {{embed: {{batch_size: 8}}}}}}
""")
    # The evaluator fixture already contains fixed references. Exercise the capture
    # review/resume lifecycle without rewriting those immutable fixture files.
    monkeypatch.setattr(module, "capture", lambda *args: 0)
    assert pdf_sweep([str(exp), "--capture-only", "--out-root", str(tmp_path / "out")]) == 0
    run = next((tmp_path / "out" / "t").iterdir())
    (run / "quality-review.json").write_text('{"status":"complete"}')
    rc = pdf_sweep([str(exp), "--resume-run", str(run)])
    out = capsys.readouterr().out
    assert rc == 0
    run = next((tmp_path / "out" / "t").iterdir())
    assert len(list((tmp_path / "out" / "t").iterdir())) == 1
    for name in ("baseline.json", "other.json", "other.compare.json", "baseline.env.json", "summary.json"):
        assert (run / name).exists(), name
    rows = json.loads((run / "summary.json").read_text())
    assert [r["variant"] for r in rows] == ["baseline", "other"] and all(r["pass"] for r in rows)
    # Synthetic fixture data is never officially comparable; the sweep says so instead of hiding it.
    assert not rows[1]["official"] and "compare:" in rows[1]["note"]
    assert "| other | ✓ |" in out


def test_pdf_manifest(tmp_path):
    pytest.importorskip("pypdfium2")
    from lab.pdf_manifest import main
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs" / "a b.pdf").write_bytes((ROOT / "eval/fixtures/synthetic.pdf").read_bytes())
    main([str(tmp_path)])
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["cases"] == [{"id": "a-b", "pdf": "pdfs/a b.pdf", "reference": "refs/a-b.json", "pages": 1}]


@pytest.mark.skipif(not os.environ.get("DOCUMENT_PIPELINE_SLOW"), reason="set DOCUMENT_PIPELINE_SLOW=1 to run the real pipeline")
def test_real_pipeline_valid_and_deterministic(tmp_path):
    import subprocess, sys
    sample = tmp_path / "sample"
    subprocess.run([sys.executable, str(ROOT / "scripts/make_sample_pdfs.py"), str(sample)], check=True)
    from eval.evaluate import check, validate_output
    from document_pipeline.pipeline import Pipeline
    pipe = Pipeline()
    for pdf in ("digital-3p", "scanned-1p"):
        a = pipe(str(sample / f"pdfs/{pdf}.pdf"))
        b = pipe(str(sample / f"pdfs/{pdf}.pdf"))
        assert validate_output(a) == []
        assert check(b, a, 0.9999)["pass"], pdf
        assert len(a["ocr_text"]) > 500


EMBED_TOK = Path(os.path.expanduser("~/huggingface/Qwen3-Embedding-0.6B"))


@pytest.mark.skipif(not EMBED_TOK.exists(), reason="needs the Qwen3-Embedding tokenizer")
def test_markdown_chunker():
    from document_pipeline.chunker import MarkdownChunker
    from document_pipeline.config import ChunkConfig
    ch = MarkdownChunker(ChunkConfig(max_tokens=40))
    text = "# Intro\n\nshort para.\n\n" + "\n\n".join(f"para {i} " + "word " * 15 for i in range(4)) \
        + "\n\n## Next\n\n" + "long " * 200 + "\n"
    chunks = ch.chunk(text)
    assert chunks == ch.chunk(text)                                   # deterministic
    assert chunks[0].startswith("# Intro")
    assert all(c.startswith("# Intro") for c in chunks if "para" in c)  # heading context carried
    assert any(c.startswith("## Next") for c in chunks)
    assert all(ch._n(c) <= 40 + 8 for c in chunks)                     # soft cap (+ separators)
    assert ch.chunk("") == [""]
