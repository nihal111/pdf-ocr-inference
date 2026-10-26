"""Local recognition admission shared by page workers; preserves response order."""
from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import Future, wait
from contextlib import asynccontextmanager
import json
import os
import socket
import struct
import threading
from uuid import uuid4

from document_pipeline.paddle_server import RegionClients


def validate_region_admission(cfg):
    policy = getattr(cfg, "paddle_region_admission", "none")
    if policy not in {"none", "round_robin", "available"}:
        raise ValueError("Unsupported region admission policy.")
    if policy == "none":
        return
    if (cfg.backend != "paddleocr" or cfg.paddle_rec_backend != "vllm-server"
            or cfg.paddle_region_dispatch != "all_gpus" or cfg.gpu_workers < 2):
        raise ValueError("Region admission requires Paddle page workers and shared vLLM recognizers.")
    if (type(cfg.paddle_region_admission_slots_per_gpu) is not int
            or cfg.paddle_region_admission_slots_per_gpu < 1
            or type(cfg.paddle_region_batch_size) is not int or cfg.paddle_region_batch_size < 1):
        raise ValueError("Region admission limits must be positive integers.")


def socket_address(path):
    if not isinstance(path, str) or not path.startswith("@inference-region-"):
        raise ValueError("Expected an internal region-admission socket.")
    return "\0" + path[1:]


def _peer_uid(writer):
    credentials = writer.get_extra_info("socket").getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    return struct.unpack("3i", credentials)[1]


class RegionAdmission:
    def __init__(self, gpus=4, workers=4, slots_per_gpu=8, slots_per_worker_gpu=4):
        if any(type(value) is not int or value < 1
               for value in (gpus, workers, slots_per_gpu, slots_per_worker_gpu)):
            raise ValueError("Admission limits must be positive integers.")
        self.gpus, self.workers = gpus, workers
        self.slots, self.worker_slots = slots_per_gpu, slots_per_worker_gpu
        self.active = [0] * gpus
        self.per_worker = [[0] * gpus for _ in range(workers)]
        self.peak = [0] * gpus
        self.worker_peak = [[0] * gpus for _ in range(workers)]
        self.grants = [0] * gpus
        self.pending = deque()
        self.next_gpu = 0
        self.closed = False
        self.handlers = set()
        if not hasattr(socket, "SO_PEERCRED"):
            raise RuntimeError("Region admission requires Linux peer credentials.")
        self.owner_uid = os.geteuid()
        # eval terminates its pipeline worker directly. Abstract sockets vanish
        # with the process; kernel peer credentials restrict admission to this UID.
        self.path = f"@inference-region-{os.getpid()}-{uuid4().hex}"
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        try:
            asyncio.run_coroutine_threadsafe(self._start(), self.loop).result(timeout=5)
        except BaseException:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
            self.loop.close()
            raise

    async def _start(self):
        self.condition = asyncio.Condition()
        self.server = await asyncio.start_unix_server(
            self._handle, socket_address(self.path), limit=256)

    def _eligible(self, worker, target=None):
        return [gpu for gpu in (range(self.gpus) if target is None else [target])
                if self.active[gpu] < self.slots and self.per_worker[worker][gpu] < self.worker_slots]

    async def _handle(self, reader, writer):
        task = asyncio.current_task()
        self.handlers.add(task)
        worker = gpu = ticket = target = None
        try:
            if self.closed or _peer_uid(writer) != self.owner_uid:
                raise ConnectionError("Admission is closed or the peer UID differs.")
            command = json.loads(await asyncio.wait_for(reader.readline(), timeout=5))
            if not isinstance(command, dict):
                raise ValueError("Admission command must be an object.")
            worker = command.get("worker")
            if type(worker) is not int or not 0 <= worker < self.workers:
                raise ValueError("Unknown page worker.")
            target = command.get("target")
            if target is not None and (type(target) is not int or not 0 <= target < self.gpus):
                raise ValueError("Unknown recognition GPU.")
            async with self.condition:
                ticket = object()
                self.pending.append((ticket, worker, target))
                while True:
                    if self.closed or reader.at_eof():
                        raise ConnectionError("Admission closed while waiting.")
                    # A worker at its local limit must not block an eligible worker behind it.
                    eligible_ticket = next((item for item, owner, destination in self.pending
                                            if self._eligible(owner, destination)), None)
                    if eligible_ticket is ticket:
                        available = self._eligible(worker, target)
                        gpu = min(available, key=lambda item: (
                            self.active[item], (item - self.next_gpu) % self.gpus))
                        self.next_gpu = (gpu + 1) % self.gpus
                        self.pending.remove((ticket, worker, target))
                        ticket = None
                        self.active[gpu] += 1
                        self.per_worker[worker][gpu] += 1
                        self.peak[gpu] = max(self.peak[gpu], self.active[gpu])
                        self.worker_peak[worker][gpu] = max(
                            self.worker_peak[worker][gpu], self.per_worker[worker][gpu])
                        self.grants[gpu] += 1
                        self.condition.notify_all()
                        break
                    try:
                        await asyncio.wait_for(self.condition.wait(), timeout=.1)
                    except asyncio.TimeoutError:
                        pass
            writer.write(json.dumps({"gpu": gpu}).encode() + b"\n")
            await writer.drain()
            message = await reader.readline()
            if message and message != b"release\n":
                raise ValueError("Unexpected admission message.")
            async with self.condition:
                self.active[gpu] -= 1
                self.per_worker[worker][gpu] -= 1
                gpu = None
                self.condition.notify_all()
            # The acknowledgment makes capacity available before the outer result completes.
            if message:
                writer.write(b"released\n")
                await writer.drain()
        except (ValueError, OSError, asyncio.TimeoutError):
            pass
        finally:
            async with self.condition:
                if ticket is not None:
                    self.pending.remove((ticket, worker, target))
                if gpu is not None:
                    self.active[gpu] -= 1
                    self.per_worker[worker][gpu] -= 1
                self.condition.notify_all()
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
            self.handlers.discard(task)

    async def _snapshot(self):
        async with self.condition:
            return {"active": list(self.active), "pending": len(self.pending),
                    "grants": list(self.grants), "peak": list(self.peak),
                    "worker_peak": [list(row) for row in self.worker_peak]}

    def snapshot(self):
        return asyncio.run_coroutine_threadsafe(self._snapshot(), self.loop).result(timeout=5)

    async def _shutdown(self):
        self.closed = True
        self.server.close()
        async with self.condition:
            self.condition.notify_all()
        await asyncio.sleep(0)
        tasks = list(self.handlers)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # wait_closed also waits for accepted connections; cancel their handlers first.
        await self.server.wait_closed()
        assert not self.pending and not any(self.active)

    def close(self):
        if not self.thread.is_alive():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self.loop).result(timeout=5)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
            self.loop.close()

    def __enter__(self):
        return self

    def __exit__(self, *ignored):
        self.close()


@asynccontextmanager
async def region_lease(path, worker, target=None):
    reader, writer = await asyncio.open_unix_connection(socket_address(path), limit=256)
    failed = False
    acquired = False
    try:
        if _peer_uid(writer) != os.geteuid():
            raise PermissionError("The admission server belongs to another UID.")
        command = {"worker": worker}
        if target is not None:
            command["target"] = target
        writer.write(json.dumps(command).encode() + b"\n")
        await writer.drain()
        line = await reader.readline()
        if not line:
            raise ConnectionError("Admission ended without granting a GPU.")
        response = json.loads(line)
        if type(response.get("gpu")) is not int or response["gpu"] < 0:
            raise ValueError("Admission returned an invalid GPU.")
        acquired = True
        yield response["gpu"]
    except BaseException:
        failed = True
        raise
    finally:
        try:
            if acquired:
                writer.write(b"release\n")
                await writer.drain()
                if await reader.readline() != b"released\n":
                    raise ConnectionError("Admission release was not acknowledged.")
        except (ConnectionError, OSError):
            if not failed:
                raise
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                if not failed:
                    raise


class AdmittedRegionClients(RegionClients):
    """Attach the actual GPU before publishing each response to Paddle's collector."""
    def __init__(self, clients, worker, path, schedule, policy="available"):
        if policy not in {"round_robin", "available"}:
            raise ValueError("Unsupported admitted routing policy.")
        super().__init__(clients, worker)
        self.worker, self.path, self.schedule = worker, path, schedule
        self.policy = policy
        self._lock = threading.Lock()
        self._pending = set()
        self._closed = False

    def create_chat_completion(self, messages, *, return_future=False, **kwargs):
        if not return_future:
            raise ValueError("Region distribution requires asynchronous requests.")
        output = Future()
        target = None
        if self.policy == "round_robin":
            target = self.next_gpu
            self.next_gpu = (target + 1) % len(self.clients)
        async def request():
            async with region_lease(self.path, self.worker, target) as gpu:
                if not 0 <= gpu < len(self.clients):
                    raise ValueError("Admission selected an unavailable recognizer.")
                output.document_pipeline_gpu = gpu
                response = self.clients[gpu].create_chat_completion(
                    messages, return_future=True, **kwargs)
                return await asyncio.wrap_future(response)
        with self._lock:
            if self._closed:
                raise RuntimeError("Region clients are closed.")
            coroutine = request()
            try:
                scheduled = self.schedule(coroutine)
            except BaseException:
                coroutine.close()
                raise
            self._pending.add(scheduled)
        def complete(done):
            with self._lock:
                self._pending.discard(done)
            if not output.set_running_or_notify_cancel():
                return
            try:
                result = done.result()
            except BaseException as error:
                output.set_exception(error)
            else:
                output.set_result(result)
        scheduled.add_done_callback(complete)
        output.add_done_callback(lambda done: scheduled.cancel() if done.cancelled() else None)
        return output

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = list(self._pending)
        for future in pending:
            future.cancel()
        wait(pending, timeout=5)
        super().close()
