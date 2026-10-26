---
title: vLLM · Server concurrency 16
description: Increase admission capacity to cover all requests allowed by the unchanged clients.
order: 14
status: passed
date: 2026-10-26
run: paddleocr_vllm_server_concurrency_16/20261026-095937
---

## Evidence and controlled change

The [previous winner's NACA profile](#/profile/db72310d9fe71f8fa226) found eight running requests per GPU and up to four waiting, with queues in 18–24% of sampled OCR time. OCR still accounts for 81.6% of the measured call.

Raise only vLLM `--max-num-seqs` to 16. Keep client concurrency four per recognizer per page worker, window 64, dynamic pages, BF16/16-chunk embeddings, four L4 GPUs, batch invariance and all frozen OCR settings. No speculative decoding or output/prefix/image-processor caching is enabled.

The previous accepted configuration has p50 3.29 s, p95 25.77 s and 26.16 pages/min. Its report is copied byte for byte for the incremental comparison. The dashboard still uses the original vLLM batch-1 baseline.

Run the same five frozen PDFs with one warm-up and five measured calls each. Accept improvement only after all frozen checks pass; repeat a control/candidate comparison if the difference is small. At most sixteen client requests per recognizer can be outstanding across the four page workers, so larger server limits would not admit more work under these client settings.

## Result

All 30 calls passed the frozen checks and all 2,790 regions ended with EOS. Performance did not improve.

| Configuration | p50 | p95 | Pages/min |
| --- | ---: | ---: | ---: |
| Previous winner · eight | 3.29 s | 25.77 s | 26.16 |
| Candidate · sixteen | 3.44 s | 26.21 s | 25.66 |

The p50 ratio against eight is 0.957×, p95 is 0.983× and throughput is 0.981×. Keep eight as the best configuration. More queued work was available, but admitting it did not produce a speedup in this trial; queue occupancy alone does not predict the result.

Next, screen embedding compilation: the current-best NACA profile still spends 14.2% of mean time in embedding. Separately profile Census's long decode before testing within-request speculative decoding. No frozen settings or references will be relaxed.
