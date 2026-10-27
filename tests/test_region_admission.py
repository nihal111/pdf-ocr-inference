import asyncio
from concurrent.futures import Future
from contextlib import AsyncExitStack
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from document_pipeline.region_admission import (
    AdmittedRegionClients, RegionAdmission, region_lease, socket_address,
    validate_region_admission,
)

# Admission authenticates peers with SO_PEERCRED, which only Linux provides.
linux_only = pytest.mark.skipif(not hasattr(socket, "SO_PEERCRED"),
                                reason="region admission requires Linux peer credentials")


def assert_socket_closed(path):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        with pytest.raises(ConnectionRefusedError):
            connection.connect(socket_address(path))


@linux_only
def test_free_gpu_receives_pending_work_and_results_keep_input_order():
    with RegionAdmission(gpus=2, workers=1, slots_per_gpu=1, slots_per_worker_gpu=1) as broker:
        assert broker.owner_uid == os.geteuid()
        async def exercise():
            release_slow = asyncio.Event()
            enough_fast = asyncio.Event()
            assignments, finished = {}, []
            async def request(index):
                async with region_lease(broker.path, 0) as gpu:
                    assignments[index] = gpu
                    if index == 0:
                        assert gpu == 0
                        await release_slow.wait()
                    else:
                        await asyncio.sleep(.002)
                        finished.append(index)
                        if len(finished) >= 4:
                            enough_fast.set()
                    return f"region-{index}"
            first = asyncio.create_task(request(0))
            while 0 not in assignments:
                await asyncio.sleep(.001)
            rest = [asyncio.create_task(request(index)) for index in range(1, 12)]
            await asyncio.wait_for(enough_fast.wait(), 2)
            assert all(assignments[index] == 1 for index in finished)
            release_slow.set()
            results = await asyncio.gather(first, *rest)
            assert results == [f"region-{index}" for index in range(12)]
        asyncio.run(exercise())
        snapshot = broker.snapshot()
        assert snapshot["active"] == [0, 0] and snapshot["pending"] == 0
        assert snapshot["peak"] == [1, 1] and sum(snapshot["grants"]) == 12
    assert_socket_closed(broker.path)


@linux_only
def test_worker_limit_does_not_block_other_workers():
    with RegionAdmission(gpus=2, workers=2, slots_per_gpu=2, slots_per_worker_gpu=1) as broker:
        async def exercise():
            async with AsyncExitStack() as stack:
                assert await stack.enter_async_context(region_lease(broker.path, 0)) == 0
                assert await stack.enter_async_context(region_lease(broker.path, 0)) == 1
                blocked_lease = region_lease(broker.path, 0)
                blocked = asyncio.create_task(blocked_lease.__aenter__())
                while broker.snapshot()["pending"] != 1:
                    await asyncio.sleep(.005)
                async with region_lease(broker.path, 1) as gpu:
                    assert gpu in (0, 1)
                    assert not blocked.done()
                blocked.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await blocked
            deadline = asyncio.get_running_loop().time() + 2
            while broker.snapshot()["pending"]:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(.01)
        asyncio.run(exercise())
        assert broker.snapshot()["active"] == [0, 0]


@linux_only
def test_concurrent_workers_respect_global_and_per_worker_limits():
    with RegionAdmission(gpus=4, workers=4, slots_per_gpu=2, slots_per_worker_gpu=1) as broker:
        async def exercise():
            async def request(index):
                async with region_lease(broker.path, index % 4):
                    await asyncio.sleep(.001 * (1 + index % 3))
            await asyncio.wait_for(asyncio.gather(*(request(i) for i in range(64))), 5)
        asyncio.run(exercise())
        snapshot = broker.snapshot()
        assert sum(snapshot["grants"]) == 64 and snapshot["active"] == [0] * 4
        assert all(peak <= 2 for peak in snapshot["peak"])
        assert all(peak <= 1 for row in snapshot["worker_peak"] for peak in row)
        assert snapshot["pending"] == 0


@linux_only
def test_request_failure_releases_capacity_without_masking_error():
    with RegionAdmission(gpus=1, workers=1, slots_per_gpu=1, slots_per_worker_gpu=1) as broker:
        async def exercise():
            with pytest.raises(RuntimeError, match="OCR request failed"):
                async with region_lease(broker.path, 0):
                    raise RuntimeError("OCR request failed")
            async with region_lease(broker.path, 0) as gpu:
                assert gpu == 0
        asyncio.run(exercise())
        assert broker.snapshot()["active"] == [0]


@linux_only
def test_disconnect_releases_active_lease():
    with RegionAdmission(gpus=1, workers=1, slots_per_gpu=1, slots_per_worker_gpu=1) as broker:
        async def exercise():
            reader, writer = await asyncio.open_unix_connection(socket_address(broker.path))
            writer.write(b'{"worker":0}\n')
            await writer.drain()
            assert await reader.readline() == b'{"gpu": 0}\n'
            writer.close()
            await writer.wait_closed()
            async with region_lease(broker.path, 0):
                pass
        asyncio.run(exercise())
        assert broker.snapshot()["active"] == [0]


@linux_only
def test_shutdown_unblocks_waiters_and_cleans_socket():
    broker = RegionAdmission(gpus=1, workers=1, slots_per_gpu=1, slots_per_worker_gpu=1)
    async def exercise():
        held = region_lease(broker.path, 0)
        await held.__aenter__()
        pending_lease = region_lease(broker.path, 0)
        pending = asyncio.create_task(pending_lease.__aenter__())
        while broker.snapshot()["pending"] != 1:
            await asyncio.sleep(.005)
        broker.close()
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(pending, 2)
        with pytest.raises(ConnectionError):
            await held.__aexit__(None, None, None)
    try:
        asyncio.run(exercise())
    finally:
        broker.close()
    assert_socket_closed(broker.path)
    assert not broker.thread.is_alive()


@linux_only
def test_paddle_collector_gets_original_order_and_actual_gpu(monkeypatch):
    from document_pipeline.paddle_server import ServerRecognition
    monkeypatch.delenv("DOCUMENT_PIPELINE_OCR_DIAGNOSTICS", raising=False)
    with RegionAdmission(gpus=2, workers=1, slots_per_gpu=1, slots_per_worker_gpu=1) as broker:
        schedule = lambda coroutine: asyncio.run_coroutine_threadsafe(coroutine, broker.loop)
        release_slow, four_fast = threading.Event(), threading.Event()
        finished = []
        class Client:
            closed = False
            def create_chat_completion(self, messages, **kwargs):
                async def response():
                    index = messages[0]
                    while index == 0 and not release_slow.is_set():
                        await asyncio.sleep(.001)
                    if index:
                        finished.append(index)
                        if len(finished) >= 4:
                            four_fast.set()
                    return SimpleNamespace(
                        choices=[SimpleNamespace(token_ids=[index + 10, 2], finish_reason="stop",
                                                 message=SimpleNamespace(content=f"region-{index}"))],
                        usage=SimpleNamespace(prompt_tokens=10))
                return schedule(response())
            def close(self):
                self.closed = True
        clients = [Client(), Client()]
        router = AdmittedRegionClients(clients, 0, broker.path, schedule)
        try:
            first = router.create_chat_completion([0], return_future=True)
            deadline = time.monotonic() + 2
            while not hasattr(first, "document_pipeline_gpu"):
                assert time.monotonic() < deadline
                time.sleep(.001)
            rest = [router.create_chat_completion([index], return_future=True) for index in range(1, 10)]
            assert four_fast.wait(2)
            assert not first.done()
            release_slow.set()
            ocr = SimpleNamespace(
                recognizer=SimpleNamespace(_doc_vlm_genai_build_request_specs=lambda *args: []),
                cfg=SimpleNamespace(require_eos=True), device="gpu:0", _local_page=1,
                stats={"batches": [], "gen_tokens": 0})
            futures = [first, *rest]
            assert ServerRecognition(ocr).collect(futures) == [f"region-{i}" for i in range(10)]
            assert ocr.stats["batches"][0]["recognizer_gpus"] == [future.document_pipeline_gpu for future in futures]
            assert all(future.document_pipeline_gpu == 1 for future in rest[:4])
            assert broker.snapshot()["active"] == [0, 0]
        finally:
            release_slow.set()
            router.close()
        assert all(client.closed for client in clients)


@linux_only
def test_client_bridge_preserves_backend_error_and_reuses_capacity():
    with RegionAdmission(gpus=1, workers=1, slots_per_gpu=1, slots_per_worker_gpu=1) as broker:
        class Client:
            def create_chat_completion(self, messages, **kwargs):
                future = Future()
                if messages == ["fail"]:
                    future.set_exception(RuntimeError("backend failed"))
                else:
                    future.set_result("ok")
                return future
            def close(self):
                pass
        router = AdmittedRegionClients([Client()], 0, broker.path,
            lambda coroutine: asyncio.run_coroutine_threadsafe(coroutine, broker.loop))
        try:
            with pytest.raises(RuntimeError, match="backend failed"):
                router.create_chat_completion(["fail"], return_future=True).result(timeout=2)
            assert router.create_chat_completion(["ok"], return_future=True).result(timeout=2) == "ok"
            assert broker.snapshot()["active"] == [0]
        finally:
            router.close()
        with pytest.raises(RuntimeError, match="closed"):
            router.create_chat_completion(["ok"], return_future=True)


@linux_only
def test_cancelling_queued_future_does_not_consume_a_gpu():
    with RegionAdmission(gpus=1, workers=1, slots_per_gpu=1, slots_per_worker_gpu=1) as broker:
        held = Future()
        class Client:
            def create_chat_completion(self, messages, **kwargs):
                return held
            def close(self):
                pass
        router = AdmittedRegionClients([Client()], 0, broker.path,
            lambda coroutine: asyncio.run_coroutine_threadsafe(coroutine, broker.loop))
        try:
            first = router.create_chat_completion([], return_future=True)
            deadline = time.monotonic() + 2
            while broker.snapshot()["active"] != [1]:
                assert time.monotonic() < deadline
                time.sleep(.001)
            queued = router.create_chat_completion([], return_future=True)
            while broker.snapshot()["pending"] != 1:
                assert time.monotonic() < deadline
                time.sleep(.001)
            assert queued.cancel()
            while broker.snapshot()["pending"]:
                assert time.monotonic() < deadline
                time.sleep(.005)
            held.set_result("finished")
            assert first.result(timeout=2) == "finished" and queued.cancelled()
            assert broker.snapshot()["grants"] == [1] and broker.snapshot()["active"] == [0]
        finally:
            router.close()


@linux_only
def test_round_robin_admission_preserves_destinations_at_the_same_capacity():
    with RegionAdmission(gpus=2, workers=1, slots_per_gpu=1, slots_per_worker_gpu=1) as broker:
        class Client:
            def create_chat_completion(self, messages, **kwargs):
                response = Future()
                response.set_result(messages[0])
                return response
            def close(self):
                pass
        router = AdmittedRegionClients([Client(), Client()], 0, broker.path,
            lambda coroutine: asyncio.run_coroutine_threadsafe(coroutine, broker.loop),
            policy="round_robin")
        try:
            futures = [router.create_chat_completion([i], return_future=True) for i in range(12)]
            assert [future.result(timeout=2) for future in futures] == list(range(12))
            assert [future.document_pipeline_gpu for future in futures] == [0, 1] * 6
            assert broker.snapshot()["peak"] == [1, 1]
            router.reset()
            future = router.create_chat_completion([12], return_future=True)
            assert future.result(timeout=2) == 12 and future.document_pipeline_gpu == 0
        finally:
            router.close()


@linux_only
def test_bound_destination_does_not_block_requests_for_other_gpu():
    with RegionAdmission(gpus=2, workers=1, slots_per_gpu=1, slots_per_worker_gpu=1) as broker:
        async def exercise():
            async with region_lease(broker.path, 0, target=0):
                blocked_lease = region_lease(broker.path, 0, target=0)
                blocked = asyncio.create_task(blocked_lease.__aenter__())
                while not broker.snapshot()["pending"]:
                    await asyncio.sleep(.001)
                async with region_lease(broker.path, 0, target=1) as gpu:
                    assert gpu == 1 and not blocked.done()
                blocked.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await blocked
        asyncio.run(exercise())


@linux_only
def test_kernel_peer_uid_is_required_before_admitting_work():
    with RegionAdmission(gpus=1, workers=1) as broker:
        broker.owner_uid = os.geteuid() + 1
        async def exercise():
            with pytest.raises(ConnectionError):
                async with region_lease(broker.path, 0):
                    pytest.fail("A different peer UID was admitted.")
        asyncio.run(exercise())
        assert broker.snapshot()["grants"] == [0]


@linux_only
def test_eval_style_termination_removes_abstract_socket():
    script = """from document_pipeline.region_admission import RegionAdmission
import signal
broker = RegionAdmission(gpus=1, workers=1)
print(broker.path, flush=True)
signal.pause()
"""
    process = subprocess.Popen([sys.executable, "-u", "-c", script],
        cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 5)[0], "Broker did not start."
        path = process.stdout.readline().strip()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            connection.connect(socket_address(path))
            connection.sendall(b'{"worker":0}\n')
            assert json.loads(connection.recv(256)) == {"gpu": 0}
            process.terminate()  # Same lifecycle as the unchanged evaluator.
            process.wait(timeout=5)
            assert connection.recv(1) == b""
        assert_socket_closed(path)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


@pytest.mark.parametrize("override", [
    {"paddle_region_admission": "invalid"},
    {"paddle_region_admission": "available", "paddle_rec_backend": "native"},
    {"paddle_region_admission": "available", "paddle_region_dispatch": "page"},
    {"paddle_region_admission": "available", "gpu_workers": 1},
    {"paddle_region_admission": "available", "paddle_region_admission_slots_per_gpu": 0},
    {"paddle_region_admission": "available", "paddle_region_admission_slots_per_gpu": True},
])
def test_invalid_admission_configuration_is_rejected(override):
    from document_pipeline.config import load_config
    cfg = load_config({"ocr": {
        "paddle_rec_backend": "vllm-server", "paddle_region_dispatch": "all_gpus",
        **override}}).ocr
    with pytest.raises(ValueError):
        validate_region_admission(cfg)
