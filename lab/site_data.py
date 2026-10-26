"""Read documentation and recorded PDF experiments without importing GPU libraries."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, median

import yaml

from document_pipeline.config import PipelineConfig
from eval.compare import compare
from eval.evaluate import load_json
from lab.run_controls import control_links

STAGES = ("render", "ocr", "chunk", "embed")
MAX_JSON = 16 * 1024 * 1024


def inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Path leaves the allowed directory")
    return resolved


def read_json(path: Path):
    if path.stat().st_size > MAX_JSON:
        raise ValueError("JSON exceeds the catalog size limit; download the artifact instead")
    return load_json(path)


def identifier(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()[:20]


def stage_metrics(rows: list[dict]) -> dict:
    """Aggregate complete, nonnegative per-call wall times; shares use means, not medians."""
    valid = []
    layouts = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("stages_s"), dict):
            continue
        names = (("prepare PDF", "render + OCR", "chunk", "embed")
                 if "render + OCR" in row["stages_s"] else tuple(STAGES))
        values = [row.get("total_s"), *(row["stages_s"].get(k) for k in names)]
        if all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values):
            valid.append(row)
            layouts.add(names)
    if not valid:
        return {"calls": 0, "stages": [], "median_total_ms": None}
    if len(layouts) != 1:
        raise ValueError("Cannot aggregate calls with different stage layouts")
    names = next(iter(layouts))
    total_mean = mean(r["total_s"] for r in valid)
    stages = []
    for name in names:
        values = [r["stages_s"][name] for r in valid]
        stages.append({"name": name, "median_ms": median(values) * 1000,
                       "mean_ms": mean(values) * 1000,
                       "share": mean(values) / total_mean if total_mean else 0})
    return {"calls": len(valid), "median_total_ms": median(r["total_s"] for r in valid) * 1000,
            "mean_total_ms": total_mean * 1000, "stages": stages,
            "dominant_stage": max(stages, key=lambda x: x["mean_ms"])["name"]}


class Catalog:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.results = self.root / "results"
        self.docs = self.root / "docs"
        self.data = self.root / "data"

    def datasets(self) -> list[dict]:
        datasets = []
        for manifest_path in sorted(self.data.glob("*/manifest.json")):
            try:
                directory = inside(self.data, manifest_path.parent)
                manifest = read_json(inside(directory, manifest_path))
                sources_path = directory / "sources.json"
                sources = read_json(inside(directory, sources_path)) if sources_path.exists() else {}
                by_id = {s["id"]: s for s in sources.get("sources", [])}
                preview_path = directory / "previews/index.json"
                previews = read_json(inside(directory, preview_path)) if preview_path.exists() else {}
                documents = []
                for case in manifest.get("cases", []):
                    pdf_root = inside(directory, directory / "pdfs")
                    pdf = inside(pdf_root, directory / case["pdf"])
                    if not pdf.is_file() or pdf.suffix != ".pdf":
                        continue
                    source = by_id.get(case["id"], {})
                    documents.append({
                        "id": case["id"], "name": pdf.name, "pages": case["pages"],
                        "title": source.get("title", case["id"]),
                        "category": source.get("category", ""),
                        "notes": source.get("selection_notes", ""),
                        "full_document": source.get("full_document") is True,
                        "source_url": source.get("source_page", ""),
                        "source_pages": source.get("selected_pages_1_based", []),
                        "sha256": source.get("pdf_sha256", ""),
                        "preview_pages": sorted(int(p) for p in previews.get(case["id"], {}).get("pages", {})),
                    })
                datasets.append({"id": directory.name, "path": str(directory.relative_to(self.root)),
                                 "purpose": manifest.get("purpose", ""), "documents": documents,
                                 "pages": sum(d["pages"] for d in documents)})
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
        return datasets

    def dataset_asset(self, dataset_id: str, document_id: str, page: int | None = None) -> Path:
        dataset = next(d for d in self.datasets() if d["id"] == dataset_id)
        document = next(d for d in dataset["documents"] if d["id"] == document_id)
        directory = inside(self.data, self.data / dataset_id)
        manifest = read_json(inside(directory, directory / "manifest.json"))
        case = next(c for c in manifest["cases"] if c["id"] == document_id)
        pdf_root = inside(directory, directory / "pdfs")
        pdf = inside(pdf_root, directory / case["pdf"])
        if page is None:
            return pdf
        if page not in document["preview_pages"] or not 1 <= page <= document["pages"]:
            raise FileNotFoundError("Page is not in the selected document")
        index = read_json(inside(directory, directory / "previews/index.json"))
        entry = index[document_id]
        if hashlib.sha256(pdf.read_bytes()).hexdigest() != entry["pdf_sha256"]:
            raise ValueError("Preview is stale; regenerate it for the current PDF")
        preview_root = inside(directory, directory / "previews")
        image = inside(preview_root, directory / entry["pages"][str(page)])
        if image.suffix != ".png" or not image.is_file():
            raise FileNotFoundError("Preview image unavailable")
        return image

    def documents(self) -> list[dict]:
        docs = []
        for directory, kind in (("pages", "guide"), ("experiments", "experiment"), ("results", "result")):
            for path in sorted((self.docs / directory).glob("*.md")):
                if path.name.startswith("_"):
                    continue
                try:
                    text = inside(self.docs, path).read_text()
                    meta = {}
                    if text.startswith("---\n"):
                        _, header, text = text.split("---", 2)
                        meta = yaml.safe_load(header) or {}
                    if not isinstance(meta, dict):
                        continue
                    docs.append({"id": f"{directory}/{path.stem}", "kind": kind,
                                 "title": str(meta.get("title", path.stem.replace("-", " ").title())),
                                 "description": str(meta.get("description", "")),
                                 "order": int(meta.get("order", 100)), "status": meta.get("status", ""),
                                 "date": str(meta.get("date", "")), "run": str(meta.get("run", "")),
                                 "profile": str(meta.get("profile", "")), "body": text.strip()})
                except (OSError, ValueError, TypeError, yaml.YAMLError):
                    continue
        return sorted(docs, key=lambda d: (d["order"], d["id"]))

    def artifact(self, relative: str) -> Path:
        path = inside(self.results, self.results / relative)
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".yaml", ".log", ".txt", ".prof", ".nsys-rep", ".ncu-rep", ".csv"}:
            raise FileNotFoundError(relative)
        return path

    def artifacts(self, directory: Path) -> list[dict]:
        entries = []
        paths = list(directory.iterdir())
        for child in paths.copy():
            if re.fullmatch(r"worker-\d+", child.name) and child.is_dir():
                try:
                    paths.extend(inside(directory, child).iterdir())
                except (OSError, ValueError):
                    continue
        for path in sorted(paths):
            try:
                relative = str(path.relative_to(self.results))
                safe = self.artifact(relative)
                entries.append({"name": str(path.relative_to(directory)), "path": relative, "bytes": safe.stat().st_size})
            except (OSError, ValueError):
                continue
        return entries

    def traces(self, path: Path, warmup: int) -> tuple[dict, list[str]]:
        if not path.exists():
            return stage_metrics([]), []
        warnings, rows, seen = [], [], {}
        try:
            path = inside(self.results, path)
            if path.stat().st_size > MAX_JSON:
                raise ValueError("Trace exceeds the catalog size limit")
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    pdf = row["pdf"]
                    seen[pdf] = seen.get(pdf, 0) + 1
                    if seen[pdf] > warmup:
                        rows.append(row)
                except (ValueError, KeyError, TypeError):
                    warnings.append("Incomplete or invalid trace row; only complete calls are shown.")
        except (OSError, ValueError) as exc:
            warnings.append(str(exc))
        return stage_metrics(rows), list(dict.fromkeys(warnings))

    def previous_comparison(self, directory: Path, variant: dict, reports: dict) -> dict:
        """Revalidate the recorded incremental comparison through the evaluator."""
        if variant.get("control_for"):
            return {"kind": "repeat"}
        name = variant["name"]
        source = None
        suffix = None
        relative = str(directory.relative_to(self.results))
        four_sources = directory / "four-request-control-sources.json"
        snapshot = directory / "previous-control.json"
        if four_sources.exists():
            source = read_json(inside(self.results, four_sources))[name]
            snapshot = directory / f"{name}.four-request-control.json"
            suffix = "vs-four-requests"
        elif snapshot.exists() and (directory / f"{name}.vs-previous.json").exists():
            source = read_json(inside(self.results, directory / "previous-control-source.json"))
            suffix = "vs-previous"
        elif variant.get("controls"):
            link = variant["controls"][0]
            if link["path"] == relative:
                control_name = link["variant"]
                previous = reports[control_name]
                snapshot = directory / f"{control_name}.json"
            else:
                source = read_json(inside(self.results, directory / "previous-control-source.json"))
            suffix = "vs-control"
        elif variant.get("previous_variant"):
            control_name = variant["previous_variant"]
            previous = reports[control_name]
            snapshot = directory / f"{control_name}.json"
        else:
            return {}
        snapshot = inside(self.results, snapshot)
        if source is not None:
            previous = read_json(snapshot)
            if hashlib.sha256(snapshot.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError("Previous-best snapshot digest mismatch")
            source_path = inside(self.results, self.root / source["path"])
        else:
            source_path = snapshot
        computed = compare(previous, reports[name])
        if suffix:
            stored = read_json(inside(self.results, directory / f"{name}.{suffix}.json"))
            if stored != computed:
                raise ValueError("Saved previous-best comparison is stale or does not match the reports")
        return {"kind": "comparison", "metrics": computed["metrics"],
                "source": {"run_id": identifier(str(source_path.parent.relative_to(self.results))),
                           "variant": source_path.stem,
                           "report_path": str(snapshot.relative_to(self.results)),
                           "created_at": previous.get("created_at", ""),
                           "fresh_control": bool(variant.get("controls")) or "/confirmation/" in str(source_path)}}

    def runs(self, detail: bool = False, run_id: str | None = None) -> list[dict]:
        runs = []
        if not self.results.exists():
            return runs
        for spec_path in sorted(self.results.glob("*/*/experiment.yaml"), reverse=True):
            directory = spec_path.parent
            relative = str(directory.relative_to(self.results))
            # Select before parsing reports, traces, or artifact inventories.
            if run_id is not None and identifier(relative) != run_id:
                continue
            try:
                spec = yaml.safe_load(inside(self.results, spec_path).read_text())
                if not isinstance(spec, dict) or "manifest" not in spec:
                    continue  # LLM serving sweeps have a different schema.
                baseline = spec.get("baseline", "baseline")
                definitions = spec.get("_expanded", spec.get("variants", {}))
                if not isinstance(definitions, dict):
                    continue
                reports, warnings = {}, []
                if spec.get("baseline_report"):
                    try:
                        snapshot = inside(self.results, directory / "baseline-report.json")
                        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != (
                                spec.get("_baseline_source") or {}).get("sha256"):
                            raise ValueError("Saved baseline snapshot digest mismatch")
                        reports[baseline] = read_json(snapshot)
                        compare(reports[baseline], reports[baseline])
                    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                        reports.pop(baseline, None)
                        warnings.append(f"Saved baseline: {exc}")
                for name in definitions:
                    if not re.fullmatch(r"[A-Za-z0-9_.=,-]+", name):
                        warnings.append(f"Unsupported variant filename: {name}")
                        continue
                    path = directory / f"{name}.json"
                    if path.exists():
                        try:
                            report = read_json(inside(self.results, path))
                            if not isinstance(report, dict):
                                raise ValueError("Report must be an object")
                            reports[name] = report
                        except (OSError, ValueError) as exc:
                            warnings.append(f"{name}: {exc}")
                variants = []
                for name, definition in definitions.items():
                    report = reports.get(name)
                    row = {"name": name, "baseline": name == baseline, "status": "pending",
                           "official": False, "speedup": None, "comparison_note": "",
                           "comparison_metrics": {}, "baseline_qualified": False,
                           "metrics": {}, "config": definition or {}, "suite_runtime": None}
                    runtime_path = directory / f"{name}.suite-runtime.json"
                    if runtime_path.exists():
                        try:
                            runtime = read_json(inside(self.results, runtime_path))
                            seconds = runtime.get("seconds")
                            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                                raise ValueError("Suite runtime must be finite nonnegative seconds")
                            row["suite_runtime"] = runtime
                        except (OSError, ValueError, AttributeError) as exc:
                            warnings.append(f"{name}: {exc}")
                    if report is not None:
                        row["status"] = "passed" if report.get("pass") is True else "failed"
                        row["metrics"] = report.get("summary") or {}
                        row["environment"] = report.get("environment") or {}
                        row["configuration"] = report.get("configuration") or {}
                        row["dataset_sha256"] = report.get("dataset_sha256")
                        row["page_count"] = sum(a.get("pages", 0) for a in report.get("assets", []))
                        row["error"] = report.get("error")
                        if name == baseline:
                            try:
                                compare(report, report)
                                row["baseline_qualified"] = True
                            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                                row["comparison_note"] = str(exc)
                        if name != baseline:
                            try:
                                computed = compare(reports.get(baseline, {}), report)
                                stored = read_json(inside(self.results, directory / f"{name}.compare.json"))
                                if stored.get("pass") is not True or stored != computed:
                                    raise ValueError("Saved comparison is missing, stale, or does not match these reports")
                                row["official"] = True
                                row["speedup"] = computed["metrics"]["p50_ms"]["speedup"]
                                row["comparison_metrics"] = computed["metrics"]
                            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                                row["comparison_note"] = str(exc)
                        if detail:
                            row["cases"] = report.get("cases") or []
                            row["cold_start"] = report.get("cold_start") or {}
                            row["timing"], trace_warnings = self.traces(
                                directory / f"{name}.trace.jsonl",
                                row["configuration"].get("warmup_per_case", 0))
                            warnings.extend(trace_warnings)
                    row.update(control_links(relative, name))
                    variants.append(row)
                statuses = {v["status"] for v in variants}
                status = "incomplete" if "pending" in statuses else "failed" if "failed" in statuses else "passed"
                lifecycle = {}
                lifecycle_path = directory / "run-status.json"
                if lifecycle_path.exists():
                    lifecycle = read_json(inside(self.results, lifecycle_path))
                    if status == "incomplete" and lifecycle.get("status") in {
                            "running", "awaiting_review", "cancelled", "failed"}:
                        status = lifecycle["status"]
                    if lifecycle.get("reason"):
                        warnings.append(str(lifecycle["reason"]))
                run = {"id": identifier(relative), "path": relative,
                       "name": spec.get("title") or spec.get("name", directory.parent.name),
                       "timestamp": directory.name, "status": status, "baseline": baseline,
                       "variants": variants, "warnings": warnings, "lifecycle": lifecycle,
                       "experiment": {"warmup": spec.get("warmup", 1),
                                      "repeats": spec.get("repeats", 5)}}
                if spec.get("baseline_report") and baseline in reports:
                    source = (self.root / spec["baseline_report"]).resolve()
                    if source.is_relative_to(self.results.resolve()):
                        run["baseline_source"] = {
                            "run_id": identifier(str(source.parent.relative_to(self.results))),
                            "variant": baseline,
                        }
                quality_path = directory / "quality-review.json"
                if quality_path.exists():
                    quality = read_json(inside(self.results, quality_path))
                    run["quality"] = quality
                    if quality.get("status") == "incomplete":
                        run["status"] = "incomplete"
                annotation_path = directory / "run-annotation.json"
                if annotation_path.exists():
                    annotation = read_json(inside(self.results, annotation_path))
                    if annotation.get("title"):
                        run["name"] = annotation["title"]
                    if annotation.get("baseline_note"):
                        run["baseline_note"] = annotation["baseline_note"]
                    run["profile_links"] = annotation.get("profile_links", [])
                    run["summary_note"] = annotation.get("summary_note", "")
                    if annotation.get("invalidation"):
                        run["invalidation"] = annotation["invalidation"]
                    for variant in variants:
                        variant["display_name"] = (annotation.get("variant_titles") or {}).get(
                            variant["name"], run["name"])
                        variant["change_description"] = (annotation.get("variant_descriptions") or {}).get(
                            variant["name"], "")
                        variant["outcome_note"] = (annotation.get("variant_outcomes") or {}).get(
                            variant["name"], {})
                    if annotation.get("status") == "superseded":
                        run["status"] = "superseded"
                        for variant in variants:
                            variant.update(official=False, speedup=None, comparison_metrics={},
                                           baseline_qualified=False)
                    if annotation.get("status") == "invalidated":
                        run["status"] = "invalidated"
                        run["invalidation"] = annotation
                        for variant in variants:
                            variant.update(official=False, speedup=None,
                                           comparison_metrics={}, baseline_qualified=False,
                                           comparison_note=annotation.get("reason", "Run invalidated"))
                progress = []
                progress_variant = lifecycle.get("variant", baseline)
                trace = directory / ("capture.trace.jsonl" if
                                     lifecycle.get("phase", "").startswith("reference capture")
                                     else f"{progress_variant}.trace.jsonl")
                if trace.exists() and trace.stat().st_size <= MAX_JSON:
                    for line in inside(self.results, trace).read_text().splitlines():
                        try:
                            row = json.loads(line)
                            if isinstance(row.get("pages"), int) and row.get("pdf"):
                                progress.append(row)
                        except (ValueError, AttributeError):
                            continue
                if progress:
                    run["progress"] = {"completed_calls": len(progress),
                                       "completed_pages": sum(row["pages"] for row in progress),
                                       "last_document": progress[-1]["pdf"]}
                for variant in variants:
                    if variant["official"]:
                        try:
                            variant["previous_comparison"] = self.previous_comparison(directory, variant, reports)
                        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                            variant["previous_comparison"] = {"error": str(exc)}
                if detail:
                    run.update({"experiment": spec, "artifacts": self.artifacts(directory)})
                runs.append(run)
            except (OSError, ValueError, TypeError, yaml.YAMLError):
                continue
        return sorted(runs, key=lambda r: r["timestamp"], reverse=True)

    def profiles(self, detail: bool = False, profile_id: str | None = None) -> list[dict]:
        profiles = []
        for path in sorted(self.results.glob("**/profile.json"), reverse=True):
            relative = str(path.parent.relative_to(self.results))
            if profile_id is not None and identifier(relative) != profile_id:
                continue
            try:
                profile = read_json(inside(self.results, path))
                if profile.get("kind") != "document-profile":
                    continue
                item = {k: profile.get(k) for k in ("label", "created_at", "pdf", "summary", "status",
                                                    "mode", "baseline_run", "variant", "error", "findings")}
                item.update({"id": identifier(relative), "path": relative})
                if detail:
                    item["record"] = profile
                    item["artifacts"] = self.artifacts(path.parent)
                    operators = path.parent / "operators.json"
                    item["operators"] = read_json(inside(self.results, operators)) if operators.exists() else []
                profiles.append(item)
            except (OSError, ValueError, TypeError, AttributeError):
                continue
        return sorted(profiles, key=lambda p: str(p.get("created_at") or p["path"]), reverse=True)

    def overview(self) -> dict:
        defaults = PipelineConfig().to_dict()
        benchmark = {}
        active = self.root / "experiments/pdf/08_vllm_five_pdf_eval.yaml"
        if not active.is_file():
            active = self.root / "experiments/pdf/07_paddleocr_vllm_gpu_distribution.yaml"
        if not active.is_file():
            active = self.root / "experiments/pdf/00_public_v3_baseline.yaml"
        if active.is_file():
            try:
                spec = yaml.safe_load(active.read_text())
                overrides = spec["base"]["pipeline"]
                for section, values in overrides.items():
                    if section in defaults and isinstance(values, dict):
                        defaults[section].update(values)
                manifest = read_json(inside(self.root, self.root / spec["manifest"]))
                benchmark = {"pdfs": len(manifest["cases"]),
                             "pages": sum(c["pages"] for c in manifest["cases"]),
                             "warmup": spec.get("warmup", 1), "repeats": spec.get("repeats", 5),
                             "hardware_note": spec.get("hardware_note", "")}
            except (OSError, KeyError, TypeError, yaml.YAMLError):
                pass
        return {"defaults": defaults, "benchmark": benchmark,
                "documents": [{k: v for k, v in d.items() if k != "body"} for d in self.documents()],
                "runs": self.runs(), "profiles": self.profiles(), "datasets": self.datasets()}
