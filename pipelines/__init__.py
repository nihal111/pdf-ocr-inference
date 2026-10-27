"""Pinned evaluator entry points: `pipelines.baseline` and `pipelines.optimized`.

Each module fixes the pipeline configuration and client environment of one measured
variant, then exposes `document_pipeline.adapter.embed_pdf`. Import only one per process.
"""
