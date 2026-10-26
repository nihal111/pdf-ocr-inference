---
title: Debugging
description: Repetition failures, the tests we ran, and what the evidence supports.
order: 6
---

This page records failures that shaped the pipeline. **EOS** is the model’s end-of-sequence marker. Reaching it shows that generation stopped normally; the transcription still needs review for missing or incorrect content.

## PaddleOCR · W-9 table repetition

**Observed:** PaddleOCR-VL-1.6 through vLLM repeated dotted filler in a first-page form/table region. It reached 8,192 tokens without EOS, so reference capture rejected it.

A **repetition penalty** discourages the model from reusing tokens. A **pixel limit** controls how much image detail reaches the reader.

| Isolated crop test | Result |
| --- | --- |
| Penalties 1.0, 1.05, 1.1 and 1.2 | Reached 8,192 tokens without EOS |
| Penalty 1.3 | Stopped after 153 tokens but omitted most of the form; rejected |
| Raise table-image limit from 1,003,520 to 2,822,400 pixels | Finished after 1,167 tokens, including later form and certification sections |
| Add 5% white padding around the crop | Finished after 1,103 tokens; retained as an alternative test |

**Decision:** use the higher table-image limit across the baseline and both optimization configurations. Other region types keep their settings; the PDF itself is unchanged.

The full five-page W-9 capture finished in **28.5 seconds**, with **all 182 regions reaching EOS**. Reference review is complete. W-9 then passed all 18 evaluation calls across the baseline and both optimization configurations (one warm-up and five measured calls per configuration) against its frozen reference. The 28.5-second figure was a preparation call; [the evaluation journal](#/doc/experiments/009-five-pdf-evaluation) records the repeated performance checks.

Related reports:

- [PaddleOCR #17737](https://github.com/PaddlePaddle/PaddleOCR/issues/17737): blank table cells triggered long, repetitive generation in VL-1.5 through vLLM.
- [PaddleOCR #18083](https://github.com/PaddlePaddle/PaddleOCR/issues/18083): dotted-line repetition in 1.5 and incomplete output in 1.6; participants suggest resolution, decoding and padding adjustments.
- [vLLM #33068](https://github.com/vllm-project/vllm/pull/33068) and duplicate [#33160](https://github.com/vllm-project/vllm/pull/33160): proposed loop detection; both closed without merging when checked.
- [PaddleX #5170](https://github.com/PaddlePaddle/PaddleX/pull/5170): proposed cleanup of repetitive output; open and unmerged when checked.

These reports motivated tests. They do not establish a universal fix or the exact cause of our failure.

## Qwen · Repetitive seaplane transcription

**Outcome:** the Qwen3-VL-2B-Instruct setup was superseded by PaddleOCR.

| Test | Saved evidence |
| --- | --- |
| Original 2,048-token run | Pages 1 and 3 did not finish; each Figure 1 and Figure 2 caption appeared 25 times |
| Confirmatory 8,192-token diagnostic | Page 2 repeated three paragraphs 11 times each and exhausted the limit; page 3 finished after 2,181 tokens; page 1 was interrupted |
| Penalties 1.05 and 1.1, with a 4,096-token limit | Page 1 kept repeating table-formatting text; neither setting was accepted |

One earlier 8,192-token failure had no saved partial text. The repetition finding comes from the later diagnostic that saved the generated text and token IDs.

Related reports: [Qwen #1611](https://github.com/QwenLM/Qwen3-VL/issues/1611) and [OCR discussion #241](https://github.com/QwenLM/Qwen3-VL/issues/241). Our tests used **greedy decoding**, which always chooses the highest-scoring next token. Qwen’s [saved generation defaults](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/generation_config.json) used sampling instead; changing decoding would require new quality and repeatability checks.

## Lessons for future debugging

- **Save raw output before cleanup.** A token-limit failure alone does not prove repetition.
- **Review coverage as well as EOS.** The 153-token W-9 output stopped while omitting content.
- **A larger token allowance can extend a loop.** More generated text does not guarantee completion.
- **Validate the complete PDF after a crop test.** Keep the fix consistent across compared configurations.

Research describes how repeated output can reinforce further repetition: [Learning to Break the Loop](https://arxiv.org/abs/2206.02369) and [Neural Text Degeneration](https://arxiv.org/abs/1904.09751). This is a plausible mechanism, not a demonstrated root cause for either case here.

External sources were checked on **26 October 2026**. [Lessons learned](#/doc/pages/lessons-learned) · [Pipeline](#/doc/pages/pipeline) · [Evaluation rules](#/doc/pages/baseline)
