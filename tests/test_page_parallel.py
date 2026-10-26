"""Page dispatch must preserve order and account for idle GPUs."""
from types import SimpleNamespace

import pytest

from document_pipeline.page_parallel import PageParallelOcr


@pytest.mark.parametrize("pages", [0, 2, 3, 5, 9])
def test_page_order_and_idle_workers(pages):
    sent = [[] for _ in range(4)]
    pool = PageParallelOcr.__new__(PageParallelOcr)
    pool._profile_next = None
    pool.connections = [SimpleNamespace(send=items.append) for items in sent]
    pool.close = lambda: None

    def receive(worker):
        request = sent[worker][0]
        return {"pages_1_based": request["pages_1_based"],
                "texts": [f"page {image}" for image in request["images"]],
                "stats": {"gen_tokens": len(request["images"]), "visual_tokens": 10 * len(request["images"])},
                "elapsed_s": 1, "peak_allocated_bytes": 100, "peak_reserved_bytes": 200}

    pool._receive = receive
    assert pool.ocr(list(range(pages))) == [f"page {i}" for i in range(pages)]
    assert pool.stats["active_workers"] == list(range(min(pages, 4)))
    assert pool.stats["gen_tokens"] == pages and pool.stats["visual_tokens"] == pages * 10
    assert len(pool.stats["workers"]) == 4
    for worker in range(4):
        if worker >= pages:
            assert not sent[worker]
        else:
            assert sent[worker][0]["images"] == list(range(worker, pages, 4))


def test_bad_worker_output_closes_pool():
    pool = PageParallelOcr.__new__(PageParallelOcr)
    pool._profile_next = None
    pool.connections = [SimpleNamespace(send=lambda request: None)]
    pool._receive = lambda worker: {"pages_1_based": [2], "texts": ["wrong page"]}
    closed = []
    pool.close = lambda: closed.append(True)
    with pytest.raises(RuntimeError, match="inconsistent pages"):
        pool.ocr(["page one"])
    assert closed == [True]


def test_dynamic_pages_follow_free_workers_and_restore_source_order():
    sent = [[] for _ in range(4)]
    pool = PageParallelOcr.__new__(PageParallelOcr)
    pool.cfg = SimpleNamespace(page_schedule="dynamic")
    pool._profile_next = None
    pool.connections = [SimpleNamespace(send=items.append) for items in sent]
    pool.close = lambda: None
    # Worker 1 finishes its first page while worker 0 is still occupied.
    ready = iter([1, 2, 1, 3, 0])
    pool._ready_worker = lambda pending: next(ready)

    def receive(worker):
        request = sent[worker][-1]
        return {"pages_1_based": request["pages_1_based"],
                "texts": [f"page {request['images'][0]}"],
                "stats": {"pages": 1, "gen_tokens": 7, "visual_tokens": 10,
                          "batches": [{"first_local_page": 1, "ended_with_eos": [True]}],
                          "page_blocks": [{"local_page": 1, "blocks": []}]},
                "elapsed_s": 1, "peak_allocated_bytes": 100, "peak_reserved_bytes": 200}

    pool._receive = receive
    assert pool.ocr(list(range(5))) == [f"page {i}" for i in range(5)]
    assert [r["pages_1_based"] for r in sent[1]] == [[2], [5]]
    assert pool.stats["workers"][0]["pages_1_based"] == [1]
    worker = pool.stats["workers"][1]
    assert worker["pages_1_based"] == [2, 5] and worker["gen_tokens"] == 14
    assert [b["first_local_page"] for b in worker["batches"]] == [1, 2]
    assert [b["local_page"] for b in worker["page_blocks"]] == [1, 2]
    assert pool.stats["gen_tokens"] == 35


def test_admission_counts_are_per_call_and_leases_are_drained():
    pool = PageParallelOcr.__new__(PageParallelOcr)
    pool.cfg = SimpleNamespace(paddle_region_admission="available",
                              paddle_region_admission_slots_per_gpu=8, paddle_region_batch_size=4)
    pool.stats = {}
    snapshots = iter([
        {"grants": [5, 7]},
        {"grants": [8, 9], "active": [0, 0], "pending": 0,
         "peak": [2, 2], "worker_peak": [[2, 1], [1, 2]]},
    ])
    pool._region_admission = SimpleNamespace(snapshot=lambda: next(snapshots))
    pool._ocr_pages = lambda images: images
    assert pool.ocr(["page"]) == ["page"]
    assert pool.stats["region_admission"]["requests_per_gpu"] == [3, 2]
    assert pool.stats["region_admission"]["pending_after_call"] == 0


def test_paddle_process_receives_only_its_current_admission_socket(monkeypatch):
    from document_pipeline.page_parallel import _PaddleProcess
    captured = []
    monkeypatch.setenv("DOCUMENT_PIPELINE_REGION_ADMISSION_SOCKET", "@inference-region-stale")
    monkeypatch.setattr("document_pipeline.page_parallel.subprocess.Popen",
                        lambda *args, **kwargs: captured.append(kwargs) or SimpleNamespace(pid=123))
    connection = SimpleNamespace(fileno=lambda: 17)
    cfg = SimpleNamespace(paddle_python="/python")
    _PaddleProcess(connection, cfg, 0, "@inference-region-current")
    _PaddleProcess(connection, cfg, 1)
    assert captured[0]["env"]["DOCUMENT_PIPELINE_REGION_ADMISSION_SOCKET"] == "@inference-region-current"
    assert "DOCUMENT_PIPELINE_REGION_ADMISSION_SOCKET" not in captured[1]["env"]
    assert captured[0]["pass_fds"] == (17,)


def test_worker_cleanup_survives_broker_close_error():
    pool = PageParallelOcr.__new__(PageParallelOcr)
    pool._closed = False
    events = []
    def fail():
        raise RuntimeError("broker close failed")
    pool._region_admission = SimpleNamespace(close=fail)
    pool.connections = [SimpleNamespace(send=lambda value: events.append("stop"),
                                        close=lambda: events.append("connection closed"))]
    alive = [True]
    pool.processes = [SimpleNamespace(
        join=lambda timeout: None, is_alive=lambda: alive[0],
        terminate=lambda: (events.append("terminated"), alive.__setitem__(0, False)),
        kill=lambda: events.append("killed"))]
    with pytest.raises(RuntimeError, match="broker close failed"):
        pool.close()
    assert events == ["stop", "terminated", "connection closed"]
