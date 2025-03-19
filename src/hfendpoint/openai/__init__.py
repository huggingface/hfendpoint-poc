from contextlib import asynccontextmanager
from typing import Any
from weakref import WeakValueDictionary

from anyio import CancelScope, create_task_group
from fastapi import Request


__services__ = WeakValueDictionary()


def register_service(name: str, service: Any):
    if name in __services__:
        raise KeyError(f"Service {name} already exists: {__services__[name]}")

    __services__[name] = service


def delete_service(name: str):
    if name in __services__:
        del __services__[name]

def get_service(name: str) -> Any:
    return __services__[name]


async def disconnection_handler(request: Request, has_disconnect: CancelScope):
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break

    has_disconnect.cancel()


@asynccontextmanager
async def scoped_cancellation_handler(request: Request) -> CancelScope:
    async with create_task_group() as group:
        group.start_soon(disconnection_handler, request, group.cancel_scope)
        yield group.cancel_scope
        group.cancel_scope.cancel()