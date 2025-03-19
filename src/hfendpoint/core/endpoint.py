import asyncio
import os
from abc import ABC
from asyncio import Future
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Protocol, TypeVar, runtime_checkable, Callable, Union

from anyio import CancelScope
from opentelemetry.trace import Tracer

from hfendpoint.core import Handler


I = TypeVar("I")
O = TypeVar("O")


@runtime_checkable
class Endpoint(Protocol[I, O]):

    def __call__(self, request: I, tracer: Tracer, is_cancelled: CancelScope) -> O:
        ...

    @property
    def handler(self) -> Handler:
        ...


class AsyncEndpoint(ABC, Endpoint[I, O]):

    __slots__ = ("_loop", "_executor")

    def __init__(self):
        super().__init__()
        self._loop = asyncio.get_event_loop()
        self._executor = self.make_executor()

    @staticmethod
    def make_executor() -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
        """
        Create a specific thread pool for this endpoint
        :return:
        """
        cores = os.sched_getaffinity(0)
        max_workers = os.environ.get("HF_ENDPOINT_MAX_WORKERS", max(2, len(cores) - 1))

        return ThreadPoolExecutor(max_workers, thread_name_prefix="hfendpoint-background-")

    async def run_in_executor(self, f: Callable, *args) -> Future:
        """
        Offload long-running computation to a background thread
        :param f: Function to call
        :param args: Parameters, as a tuple, to feed to the function
        :return:
        """
        return await self._loop.run_in_executor(self._executor, f, *args)

