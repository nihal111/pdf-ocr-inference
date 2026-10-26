---
title: The benchmark PDFs
description: Five fixed PDFs, 20 pages, and a mix of document layouts.
order: 3
---

**Five PDFs · 20 pages · at most five pages each.** Every experiment uses these same inputs to compare performance across different document layouts.

## What’s included

Open a document to preview its pages, download the benchmark PDF and find its publisher link.

| Document | Pages | Reading challenges |
| --- | ---: | --- |
| [English earthquake checklist](#/dataset/public-v3/earthquake-checklist-en/1) | 2 | Short instructions, columns and illustrations |
| [Seaplane research report](#/dataset/public-v3/naca-seaplanes/1) | 3 | Historical scan, equations and graphs |
| [IRS Form W-9](#/dataset/public-v3/irs-w9/1) | 5 | Blank fields, checkboxes and form tables |
| [U.S. Census income report](#/dataset/public-v3/census-income/1) | 5 | Dense text, charts and a sideways table |
| [Attention Is All You Need](#/dataset/public-v3/attention-paper/1) | 5 | Research text, equations and diagrams |

The checklist and seaplane report are complete PDFs. The other documents contain five selected source pages each; those pages retain their original layout.

## What “frozen” means

- **Inputs stay fixed:** the same PDF files and selected pages are used throughout a comparison.
- **References stay fixed:** saved baseline text and embeddings are the expected outputs for later runs.
- **Changes are detectable:** file hashes act as fingerprints, identifying any change to the inputs or references.

Files live under `data/public-v3/`. `manifest.json` lists the inputs and references; `dataset-freeze.json` records their fingerprints; `sources.json` records publisher attribution and original page selections.

## What this sample tells us

The selection keeps experiments manageable while covering several layouts. Its results describe performance on these five PDFs; broader workloads and OCR accuracy need separate evaluation.

[Measurement method and pass criteria](#/doc/pages/baseline) · [Browse all five PDFs](#/datasets)
