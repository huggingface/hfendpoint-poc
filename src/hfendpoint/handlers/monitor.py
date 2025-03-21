import asyncio
from abc import ABC
from typing import AsyncIterable

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, NonNegativeInt


class MonitoringStats(BaseModel):
    in_flight: NonNegativeInt
    in_queue: NonNegativeInt
    max_in_flight: NonNegativeInt


class EngineMonitor(ABC):

    __slots__ = ("_queue", "_current")

    def __init__(self):
        self._queue = asyncio.LifoQueue(1)
        self._current = MonitoringStats(in_flight=0, in_queue=0, max_in_flight=0)

        logger.info("Initialized EngineMonitor for Horizontal Pods Autoscaler")

    async def broadcast(self, message: MonitoringStats):
        logger.debug(f"Broadcasting HPA message: {message}")
        self._current = message

        if self._queue.full():
            logger.warning(f"Purging monitoring queue (size={self._queue.maxsize})")
            self._queue.get_nowait()

        await self._queue.put(message)

    def broadcast_sync(self, message: MonitoringStats):
        asyncio.create_task(self.broadcast(message))

    async def recv(self, request: Request) -> AsyncIterable[str]:
        def _to_sse_event(stats: MonitoringStats) -> str:
            return f"event: engine_state_event\ndata: {stats.model_dump_json()}\n\n"

        yield _to_sse_event(self._current)
        while not await request.is_disconnected():
            msg: MonitoringStats = await self._queue.get()
            yield _to_sse_event(msg)


def with_engine_monitor_ws(monitor: EngineMonitor) -> APIRouter:
    router = APIRouter(tags=["hfendpoint", "proxy", "hpa"])

    @router.get("/state")
    async def __hf_proxy_monitor__(request: Request) -> StreamingResponse:
        return StreamingResponse(monitor.recv(request), media_type="text/event-stream")

    return router