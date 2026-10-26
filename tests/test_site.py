"""Catalog integrity, profile persistence and the loopback HTTP boundary."""
import http.client
import hashlib
import json
import shutil
import threading
from pathlib import Path

import pytest
import yaml

from eval.compare import compare
from lab.site import make_server
from lab.site_data import Catalog, identifier


def report(latency=100):
    return {"schema_version": 1, "pass": True, "dataset_sha256": "fixed",
            "configuration": {"warmup_per_case": 1, "repeats_per_case": 5, "minimum_cosine": .99},
            "environment": {"hardware_id": "gpu-a", "gpu_model": "L4", "gpu_count": 1,
                            "driver_version": "test", "concurrency": 1},
            "summary": {"comparable": True, "p50_ms": latency, "p95_ms": latency * 1.1,
                        "pages_per_minute_sequential": 60000 / latency}, "cases": []}


def sweep(root):
    directory = root / "results/pdf_test/20261026-120000"
    directory.mkdir(parents=True)
    (directory / "experiment.yaml").write_text(yaml.safe_dump(
        {"name": "PDF test", "manifest": "manifest.json", "variants": {"baseline": {}, "candidate": {}}}))
    base, candidate = report(), report(50)
    for name, value in [("baseline.json", base), ("candidate.json", candidate),
                        ("candidate.compare.json", compare(base, candidate))]:
        (directory / name).write_text(json.dumps(value))
    return directory


def test_catalog_requires_saved_and_revalidated_comparison(tmp_path):
    directory = sweep(tmp_path)
    candidate = Catalog(tmp_path).runs(True)[0]["variants"][1]
    assert candidate["official"] and candidate["speedup"] == 2
    assert candidate["comparison_metrics"]["p95_ms"]["speedup"] == 2
    assert candidate["comparison_metrics"]["pages_per_minute_sequential"]["speedup"] == 2
    assert Catalog(tmp_path).runs()[0]["variants"][0]["baseline_qualified"]
    stale = report(40)
    (directory / "candidate.json").write_text(json.dumps(stale))
    candidate = Catalog(tmp_path).runs(True)[0]["variants"][1]
    assert not candidate["official"] and candidate["speedup"] is None
    assert candidate["comparison_metrics"] == {}
    assert "stale" in candidate["comparison_note"]
    (directory / "candidate.json").write_text("{")
    run = Catalog(tmp_path).runs()[0]
    assert run["status"] == "incomplete" and run["warnings"]


def test_catalog_does_not_promote_invalid_reports(tmp_path):
    directory = sweep(tmp_path)
    invalid = report(50)
    invalid["configuration"]["repeats_per_case"] = 1
    (directory / "candidate.json").write_text(json.dumps(invalid))
    candidate = Catalog(tmp_path).runs()[0]["variants"][1]
    assert not candidate["official"] and "five measured" in candidate["comparison_note"]


def test_selected_run_reads_only_its_own_files_and_revalidates_changes(tmp_path, monkeypatch):
    directory = sweep(tmp_path)
    unrelated = tmp_path / "results/unrelated/20261026-120001"
    shutil.copytree(directory, unrelated)
    catalog = Catalog(tmp_path)
    run_id = identifier(str(directory.relative_to(catalog.results)))
    expected = next(run for run in catalog.runs(True) if run["id"] == run_id)
    original_read_text, original_read_bytes = Path.read_text, Path.read_bytes

    def read_text(path, *args, **kwargs):
        assert not path.is_relative_to(unrelated), "Read an unrelated run"
        return original_read_text(path, *args, **kwargs)

    def read_bytes(path, *args, **kwargs):
        assert not path.is_relative_to(unrelated), "Read an unrelated run"
        return original_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    assert catalog.runs(True, run_id=run_id) == [expected]
    assert catalog.runs(True, run_id="missing") == []
    # A selected detail still checks the current report against its saved comparison.
    (directory / "candidate.json").write_text(json.dumps(report(40)))
    variant = catalog.runs(True, run_id=run_id)[0]["variants"][1]
    assert not variant["official"] and "stale" in variant["comparison_note"]


def test_selected_profile_does_not_read_other_profiles(tmp_path, monkeypatch):
    import lab.site_data as site_data

    for name in ("selected", "unrelated"):
        directory = tmp_path / "results/profiles" / name
        directory.mkdir(parents=True)
        (directory / "profile.json").write_text(json.dumps({
            "kind": "document-profile", "status": "complete", "label": name,
            "summary": {"median_total_ms": 100}, "created_at": name}))
        (directory / "operators.json").write_text("[]")
    catalog = Catalog(tmp_path)
    expected = next(p for p in catalog.profiles(True) if p["label"] == "selected")
    original_read = site_data.read_json

    def read_selected(path):
        assert "unrelated" not in path.parts, "Read another profile's record or operators"
        return original_read(path)

    monkeypatch.setattr(site_data, "read_json", read_selected)
    assert catalog.profiles(True, profile_id=expected["id"]) == [expected]
    assert catalog.profiles(True, profile_id="missing") == []


def test_run_summary_keeps_measurement_settings_without_detailed_traces(tmp_path, monkeypatch):
    directory = sweep(tmp_path)
    spec = yaml.safe_load((directory / "experiment.yaml").read_text())
    spec.update(warmup=2, repeats=7)
    (directory / "experiment.yaml").write_text(yaml.safe_dump(spec))

    def no_stage_aggregation(*args, **kwargs):
        pytest.fail("Summary requested detailed stage timings")

    monkeypatch.setattr(Catalog, "traces", no_stage_aggregation)
    run = Catalog(tmp_path).runs()[0]
    assert run["experiment"] == {"warmup": 2, "repeats": 7}
    assert "artifacts" not in run
    assert all("cases" not in variant and "timing" not in variant for variant in run["variants"])


def test_saved_baseline_is_verified_without_adding_a_duplicate_run_row(tmp_path):
    directory = sweep(tmp_path)
    base_bytes = (directory / "baseline.json").read_bytes()
    (directory / "baseline-report.json").write_bytes(base_bytes)
    spec = yaml.safe_load((directory / "experiment.yaml").read_text())
    spec.update(baseline_report="results/original/baseline.json",
                _baseline_source={"sha256": hashlib.sha256(base_bytes).hexdigest()})
    del spec["variants"]["baseline"]
    (directory / "experiment.yaml").write_text(yaml.safe_dump(spec))
    run = Catalog(tmp_path).runs()[0]
    assert run["baseline_source"] == {
        "run_id": identifier("original"), "variant": "baseline"}
    variants = run["variants"]
    assert len(variants) == 1 and variants[0]["name"] == "candidate"
    assert variants[0]["official"] and variants[0]["speedup"] == 2
    (directory / "baseline-report.json").write_text(json.dumps(report(200)))
    run = Catalog(tmp_path).runs()[0]
    assert "baseline_source" not in run
    assert not run["variants"][0]["official"]
    assert any("digest mismatch" in warning for warning in run["warnings"])


def test_suite_runtime_is_separate_from_eval_metrics(tmp_path):
    directory = sweep(tmp_path)
    path = directory / "candidate.suite-runtime.json"
    original = (directory / "candidate.json").read_bytes()
    path.write_text(json.dumps({"seconds": 483, "source": "sweep_log_rounded_seconds"}))
    variants = Catalog(tmp_path).runs()[0]["variants"]
    assert variants[0]["suite_runtime"] is None
    assert variants[1]["suite_runtime"]["seconds"] == 483
    assert variants[1]["official"] and variants[1]["speedup"] == 2
    path.write_text('{"seconds": -1}')
    run = Catalog(tmp_path).runs()[0]
    assert run["variants"][1]["suite_runtime"] is None and run["warnings"]
    assert (directory / "candidate.json").read_bytes() == original


def test_previous_best_comparison_revalidates_snapshot_and_saved_metrics(tmp_path):
    directory = sweep(tmp_path)
    previous = report(80)
    snapshot = json.dumps(previous).encode()
    (directory / "previous-control.json").write_bytes(snapshot)
    source = {"path": "results/prior/20261026-110000/prior.json",
              "sha256": hashlib.sha256(snapshot).hexdigest()}
    (directory / "previous-control-source.json").write_text(json.dumps(source))
    comparison_path = directory / "candidate.vs-previous.json"
    comparison_path.write_text(json.dumps(compare(previous, report(50))))
    catalog = Catalog(tmp_path)
    candidate = catalog.runs()[0]["variants"][1]
    assert candidate["previous_comparison"]["metrics"]["p50_ms"]["speedup"] == 1.6
    assert candidate["speedup"] == 2  # Original baseline comparison remains distinct.
    assert candidate["previous_comparison"]["source"]["run_id"] == identifier("prior/20261026-110000")
    assert catalog.runs(True)[0]["variants"][1]["previous_comparison"] == candidate["previous_comparison"]
    comparison_path.write_text(json.dumps(compare(previous, report(40))))
    assert "stale" in catalog.runs()[0]["variants"][1]["previous_comparison"]["error"]
    (directory / "previous-control.json").write_text(json.dumps(report(90)))
    candidate = catalog.runs()[0]["variants"][1]
    assert "digest mismatch" in candidate["previous_comparison"]["error"]
    assert candidate["speedup"] == 2


@pytest.mark.parametrize("change", ["dataset", "hardware", "failed", "superseded", "escape"])
def test_previous_best_rejects_incompatible_or_excluded_evidence(tmp_path, change):
    directory = sweep(tmp_path)
    previous = report(80)
    if change == "dataset":
        previous["dataset_sha256"] = "different"
    elif change == "hardware":
        previous["environment"]["hardware_id"] = "other"
    elif change == "failed":
        previous["pass"] = False
    elif change == "superseded":
        (directory / "run-annotation.json").write_text('{"status":"superseded"}')
    snapshot = json.dumps(previous).encode()
    (directory / "previous-control.json").write_bytes(snapshot)
    (directory / "previous-control-source.json").write_text(json.dumps({
        "path": "../outside.json" if change == "escape" else "results/prior/20261026-110000/prior.json",
        "sha256": hashlib.sha256(snapshot).hexdigest()}))
    (directory / "candidate.vs-previous.json").write_text("{}")
    candidate = Catalog(tmp_path).runs()[0]["variants"][1]
    assert not candidate.get("previous_comparison", {}).get("metrics")
    if change != "superseded":
        assert candidate["previous_comparison"]["error"]


def test_invalidated_run_retains_original_reports_but_cannot_claim_speedup(tmp_path):
    directory = sweep(tmp_path)
    original = (directory / "baseline.json").read_bytes()
    (directory / "run-annotation.json").write_text(json.dumps({
        "title": "Qwen3-VL · invalidated baseline", "status": "invalidated",
        "reason": "OCR reached the token limit without finishing."}))
    run = Catalog(tmp_path).runs(True)[0]
    assert run["status"] == "invalidated" and run["name"].startswith("Qwen3-VL")
    assert run["invalidation"]["reason"]
    assert all(not v["official"] and v["speedup"] is None for v in run["variants"])
    assert all(v["status"] == "passed" for v in run["variants"])
    assert (directory / "baseline.json").read_bytes() == original


def test_superseded_run_preserves_quality_reason_and_excludes_comparisons(tmp_path):
    directory = sweep(tmp_path)
    (directory / "run-annotation.json").write_text(json.dumps({
        "status": "superseded", "invalidation": {"reason": "Incomplete repetitive OCR."},
        "profile_links": [{"path": "profiles/earlier", "variant": "candidate"}]}))
    run = Catalog(tmp_path).runs(True)[0]
    assert run["status"] == "superseded" and run["invalidation"]["reason"]
    assert run["profile_links"][0]["path"] == "profiles/earlier"
    assert all(not v["official"] and not v["baseline_qualified"]
               and not v["comparison_metrics"] for v in run["variants"])


def test_run_progress_and_cancellation_do_not_imply_passing(tmp_path):
    directory = tmp_path / "results/full/20261026-034106"
    directory.mkdir(parents=True)
    (directory / "experiment.yaml").write_text(
        "name: Full baseline\nmanifest: data/full/manifest.json\nvariants:\n  baseline: {}\n")
    lifecycle = directory / "run-status.json"
    lifecycle.write_text(json.dumps({"status": "running", "phase": "reference capture"}))
    (directory / "capture.trace.jsonl").write_text(
        '{"pdf":"report.pdf","pages":48}\n{"pdf":')
    run = Catalog(tmp_path).runs()[0]
    assert run["status"] == "running" and run["progress"]["completed_pages"] == 48
    assert run["progress"]["completed_calls"] == 1
    assert run["variants"][0]["metrics"] == {} and not run["variants"][0]["official"]
    lifecycle.write_text(json.dumps({"status": "awaiting_review", "phase": "reference capture finished"}))
    run = Catalog(tmp_path).runs()[0]
    assert run["status"] == "awaiting_review" and run["progress"]["completed_pages"] == 48
    assert run["variants"][0]["metrics"] == {}
    lifecycle.write_text(json.dumps({"status": "cancelled", "phase": "superseded", "reason": "Dataset replaced"}))
    run = Catalog(tmp_path).runs()[0]
    assert run["status"] == "cancelled" and "Dataset replaced" in run["warnings"]
    assert not run["variants"][0]["official"]


def test_trace_warmup_and_incomplete_row(tmp_path):
    directory = sweep(tmp_path)
    rows = [{"pdf": "a.pdf", "total_s": n, "stages_s": {"render": 0, "ocr": n, "chunk": 0, "embed": 0}}
            for n in (100, 1, 3)]
    path = directory / "baseline.trace.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"pdf":')
    summary, warnings = Catalog(tmp_path).traces(path, warmup=1)
    assert summary["calls"] == 2 and summary["median_total_ms"] == 2000
    assert summary["dominant_stage"] == "ocr" and warnings


def test_combined_render_ocr_trace_keeps_complete_calls_and_excludes_warmup(tmp_path):
    directory = sweep(tmp_path)
    path = directory / "baseline.trace.jsonl"
    rows = [
        {"pdf": "a.pdf", "total_s": total,
         "stages_s": {"prepare PDF": 0, "render + OCR": total - .2, "chunk": .1, "embed": .1}}
        for total in (100, 2, 4)
    ]
    rows += [{"pdf": "a.pdf", "total_s": 999, "stages_s": {"render + OCR": 999}},
             {"pdf": "a.pdf", "total_s": 3,
              "stages_s": {"prepare PDF": 0, "render + OCR": float("nan"), "chunk": .1, "embed": .1}}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary, _ = Catalog(tmp_path).traces(path, warmup=1)
    assert summary["calls"] == 2
    assert summary["median_total_ms"] == 3000
    assert summary["dominant_stage"] == "render + OCR"
    combined = next(stage for stage in summary["stages"] if stage["name"] == "render + OCR")
    assert combined["median_ms"] == pytest.approx(2800)
    assert combined["share"] == pytest.approx(2.8 / 3)


def test_stage_metrics_reject_mixed_layouts():
    from lab.site_data import stage_metrics
    rows = [
        {"total_s": 1, "stages_s": {"render": .1, "ocr": .7, "chunk": .1, "embed": .1}},
        {"total_s": 1, "stages_s": {"prepare PDF": .1, "render + OCR": .7, "chunk": .1, "embed": .1}},
    ]
    with pytest.raises(ValueError, match="different stage layouts"):
        stage_metrics(rows)


def test_artifact_paths_cannot_escape(tmp_path):
    directory = sweep(tmp_path)
    secret = tmp_path / "outside.json"
    secret.write_text('{"secret": true}')
    (directory / "linked.json").symlink_to(secret)
    catalog = Catalog(tmp_path)
    for path in ("../outside.json", "/etc/passwd", str((directory / "linked.json").relative_to(catalog.results))):
        with pytest.raises((ValueError, FileNotFoundError)):
            catalog.artifact(path)


def test_dataset_previews_are_manifest_scoped_and_fresh(tmp_path):
    directory = tmp_path / "data/public-v1"
    (directory / "pdfs").mkdir(parents=True)
    (directory / "previews").mkdir()
    pdf = directory / "pdfs/example.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    image = directory / "previews/example.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    (directory / "manifest.json").write_text(json.dumps({"cases": [
        {"id": "example", "pdf": "pdfs/example.pdf", "pages": 1}]}))
    (directory / "previews/index.json").write_text(json.dumps({"example": {
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "pages": {"1": "previews/example.png"}}}))
    catalog = Catalog(tmp_path)
    assert catalog.datasets()[0]["pages"] == 1
    assert catalog.dataset_asset("public-v1", "example") == pdf
    assert catalog.dataset_asset("public-v1", "example", 1) == image
    with pytest.raises(FileNotFoundError):
        catalog.dataset_asset("public-v1", "example", 2)
    with pytest.raises(StopIteration):
        catalog.dataset_asset("public-v1", "../manifest")
    pdf.write_bytes(b"%PDF-1.4\nchanged")
    with pytest.raises(ValueError, match="stale"):
        catalog.dataset_asset("public-v1", "example", 1)


def test_http_static_files_local_origin_and_escaped_markdown(tmp_path):
    from lab.export_site import export_site

    directory = sweep(tmp_path)
    run_id = identifier(str(directory.relative_to(tmp_path / "results")))
    profile = tmp_path / "results/profiles/selected"
    profile.mkdir(parents=True)
    (profile / "profile.json").write_text(json.dumps({"kind": "document-profile"}))
    profile_id = identifier("profiles/selected")
    docs = tmp_path / "docs/pages"
    docs.mkdir(parents=True)
    (docs / "example.md").write_text("---\ntitle: Example\n---\n<script>alert(1)</script>")
    shutil.copytree(Path(__file__).resolve().parents[1] / "web", tmp_path / "web")
    export_site(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("not public")
    (tmp_path / "site/escape.txt").symlink_to(outside)
    try:
        server = make_server(tmp_path / "site", 0)
    except PermissionError:
        pytest.skip("HTTP test needs a host execution context that can bind loopback")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
    try:
        assert server.server_address[0] == "127.0.0.1"
        connection.request("GET", "/site-data/overview.json")
        response = connection.getresponse()
        assert response.status == 200 and json.loads(response.read())["runs"]
        for kind, selected in (("runs", run_id), ("profiles", profile_id)):
            connection.request("GET", f"/site-data/{kind}/{selected}.json")
            response = connection.getresponse()
            assert response.status == 200 and json.loads(response.read())["id"] == selected
        for headers in ({"Host": "attacker.example"}, {"Origin": "https://attacker.example"}):
            connection.request("GET", "/site-data/overview.json", headers=headers)
            response = connection.getresponse()
            assert response.status == 403
            response.read()
        for path in ("/api/overview", "/artifact?path=../../pyproject.toml", "/escape.txt",
                     "/%2e%2e/outside.txt", "/site-data/"):
            connection.request("GET", path)
            response = connection.getresponse()
            assert response.status == 404
            response.read()
        connection.request("GET", "/site-data/docs/pages/example.json")
        response = connection.getresponse()
        body = json.loads(response.read())
        assert "&lt;script&gt;" in body["html"] and "<script>" not in body["html"]
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("placement", ["parent", "page_workers"])
def test_profile_preserves_measured_calls_and_configuration(tmp_path, monkeypatch, placement):
    from lab.pdf_prof import main
    import document_pipeline.pipeline as pipeline
    import lab.pdf_env as environment
    calls = []
    closed = []

    class FakePipeline:
        init_s = .2
        trace_path = None

        def __init__(self, cfg):
            self.cfg = cfg

        def __call__(self, path):
            calls.append(path)
            if self.trace_path:
                stages = ({"prepare PDF": .1, "render + OCR": 1.5, "chunk": .1, "embed": .3}
                          if self.cfg.render.placement == "page_workers" else
                          {"render": .1, "ocr": 1.5, "chunk": .1, "embed": .3})
                with open(self.trace_path, "a") as stream:
                    stream.write(json.dumps({"pdf": "x.pdf", "pages": 1, "chunks": 2, "total_s": 2,
                                             "stages_s": stages,
                                             "ocr_stats": {"gen_tokens": 150}}) + "\n")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(pipeline, "Pipeline", FakePipeline)
    monkeypatch.setattr(environment, "collect", lambda cfg: {"hardware_id": "test"})
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    out = tmp_path / "results/profiles"
    main([str(pdf), "--out", str(out), "--calls", "2", "--warmup", "1", "--no-torch-profile",
          "--config", json.dumps({"render": {"placement": placement}}),
          "--baseline-run", "pdf_test/20261026-120000"])
    record_path = next(out.glob("*/profile.json"))
    record = json.loads(record_path.read_text())
    assert len(calls) == 3 and record["summary"]["calls"] == 2
    assert closed == [True]
    assert record["ocr_timing_stage"] == ("render + OCR" if placement == "page_workers" else "ocr")
    assert record["summary"]["generated_tok_s"] == 100
    assert record["config"]["ocr"]["page_batch_size"] == 1
    assert record["status"] == "complete" and record["pdf"]["sha256"]
    assert len((record_path.parent / "stages.jsonl").read_text().splitlines()) == 2
    assert Catalog(tmp_path).profiles(True)[0]["record"]["baseline_run"] == "pdf_test/20261026-120000"
