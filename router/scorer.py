"""Micro-batching scorer: concurrent /route calls are coalesced into one forward pass.

A PassNet forward pass costs nearly the same for 1 request as for 64 (fixed per-call overhead
dominates), so under load requests that arrive while a pass is running are scored together in the
next one. Nothing waits at low load. With batching disabled every request runs its own pass.
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from router.model import PassNet, collate


class Scorer:
    def __init__(self, net: PassNet, version: str, max_batch: int = 64, max_wait_ms: float = 0.0, batching: bool = True):
        self.net, self.version = net, version
        self.max_batch, self.max_wait = max_batch, max_wait_ms / 1000
        self.batching = batching
        self.q: asyncio.Queue | None = None
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="scorer")
        self.lock = threading.Lock()
        self.batches = 0
        self.items = 0
        self._task = None

    def swap(self, net: PassNet, version: str):
        with self.lock:
            self.net, self.version = net, version

    def _forward(self, reqs: list[dict]) -> tuple[np.ndarray, str]:
        with self.lock, torch.inference_mode():
            z = self.net.logits(*collate(reqs), calibrated=True).numpy()
            return z, self.version

    async def start(self):
        self.q = asyncio.Queue()
        if self.batching:
            self._task = asyncio.create_task(self._loop())

    async def score(self, req: dict) -> tuple[np.ndarray, str]:
        """-> (M, B) calibrated logits for every catalog model, and the model version used."""
        loop = asyncio.get_running_loop()
        if not self.batching:
            # a ~0.1 ms forward pass is cheaper to run inline than to hand to a thread (GIL ping-pong)
            z, v = self._forward([req])
            return z[0], v
        fut = loop.create_future()
        await self.q.put((req, fut))
        return await fut

    async def _loop(self):
        """Natural batching: while one forward pass runs on the scorer thread, new requests queue up,
        and the next pass takes all of them. Optional max_wait adds a short collection window."""
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.q.get()]
            if self.max_wait > 0 and self.q.qsize() < self.max_batch:
                await asyncio.sleep(self.max_wait)
            while len(batch) < self.max_batch and not self.q.empty():
                batch.append(self.q.get_nowait())
            try:
                z, v = await loop.run_in_executor(self.pool, self._forward, [b[0] for b in batch])
                for i, (_, f) in enumerate(batch):
                    if not f.done():
                        f.set_result((z[i], v))
            except Exception as e:  # never strand callers
                for _, f in batch:
                    if not f.done():
                        f.set_exception(e)
            self.batches += 1
            self.items += len(batch)
