"""Presentation links for recorded control measurements and their candidates.

Keep report contents and the original evaluator baseline untouched. Explicit links
avoid classifying a new configuration as a repeat based on its display name.
"""

CONTROL_GROUPS = [
    (
        ("paddleocr_vllm_embedding_precision/20261026-085622", "vllm_batch_4_all_gpus"),
        [("paddleocr_vllm_embedding_precision/20261026-085622", "embed_bf16_b16")],
    ),
    (
        ("paddleocr_vllm_region_scheduling/20261026-090936", "embed_bf16_b16"),
        [("paddleocr_vllm_region_scheduling/20261026-090936", name)
         for name in ("region_window_64", "dynamic_pages")],
    ),
    (
        ("paddleocr_vllm_combined_scheduling/20261026-092829", "region_window_64"),
        [("paddleocr_vllm_combined_scheduling/20261026-092829", "combined_scheduling")],
    ),
    (
        ("paddleocr_vllm_compiled_embeddings/20261026-101652", "server_8_eager_control"),
        [("paddleocr_vllm_compiled_embeddings/20261026-101652", "server_8_compiled_embeddings")],
    ),
    (
        ("paddleocr_vllm_region_admission/20261026-113511", "eager_round_robin"),
        [("paddleocr_vllm_region_admission/20261026-113511", name)
         for name in ("admitted_round_robin_8", "available_gpu_8")],
    ),
    (
        ("paddleocr_vllm_admission_capacity/20261026-115305", "available_gpu_8_control"),
        [("paddleocr_vllm_admission_capacity/20261026-115305", name)
         for name in ("available_gpu_4", "available_gpu_12")],
    ),
    (
        ("paddleocr_vllm_decoder_tiles/20261026-122554", "decoder_tiles_control"),
        [("paddleocr_vllm_decoder_tiles/20261026-123223", "decoder_tiles_16_32"),
         ("paddleocr_vllm_fixed_decoder_tiles/20261026-124322", "decoder_tiles_fixed32")],
    ),
    (
        ("paddleocr_vllm_decoder_columns/20261026-125752", "decoder_columns_control"),
        [("paddleocr_vllm_decoder_columns/20261026-130349", "decoder_columns_shaped")],
    ),
    (
        ("paddleocr_vllm_parallel_embeddings/20261026-133651", "embedding_single_gpu_control"),
        [("paddleocr_vllm_parallel_embeddings/20261026-134054", "embedding_four_gpus")],
    ),
    (
        ("paddleocr_vllm_worker_render/20261026-140633", "render_parent_control"),
        [("paddleocr_vllm_worker_render/20261026-141545", "render_in_page_workers")],
    ),
    (
        ("paddleocr_vllm_three_draft_tiles/20261026-144303", "draft_4_control"),
        [("paddleocr_vllm_three_draft_tiles/20261026-145126", "draft_3_tiles32")],
    ),
    (
        ("paddleocr_vllm_six_request_tiles/20261026-150343", "server_8_draft4_control"),
        [("paddleocr_vllm_six_request_tiles/20261026-151300", "server_6_draft4_tiles32")],
    ),
]

# The final control is also the latest verification of the retained configuration.
RETAINED_RUN = ("paddleocr_vllm_six_request_tiles/20261026-150343", "server_8_draft4_control")
SLOWER_FOLLOWUP = ("paddleocr_vllm_six_request_tiles/20261026-151300", "server_6_draft4_tiles32")


def control_links(path: str, variant: str) -> dict:
    def link(key):
        return {"path": key[0], "variant": key[1]}

    key = (path, variant)
    controls = [link(control) for control, candidates in CONTROL_GROUPS if key in candidates]
    candidates = [link(candidate) for control, group in CONTROL_GROUPS
                  if key == control for candidate in group]
    presentation = {}
    if key == RETAINED_RUN:
        presentation = {
            "title": "PaddleOCR-VL-1.6 · vLLM · Four drafts, eight requests/GPU · Retained configuration",
            "description": "Latest verification: render in OCR workers, embeddings across four GPUs, fixed 32-row decoder tiles.",
            "supporting_experiments": [link(SLOWER_FOLLOWUP)],
        }
    elif key == SLOWER_FOLLOWUP:
        presentation = {"supporting_for": [link(RETAINED_RUN)]}
    initial_previous = {
        "vllm_batch_4": "vllm_batch_1",
        "vllm_batch_4_all_gpus": "vllm_batch_4",
    } if path == "paddleocr_vllm_five_pdf_eval/20261026-081822" else {}
    return {**({"controls": controls} if controls else {}),
            **({"control_for": candidates} if candidates else {}),
            **({"previous_variant": initial_previous[variant]} if variant in initial_previous else {}),
            **({"presentation": presentation} if presentation else {})}
