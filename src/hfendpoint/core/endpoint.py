from typing import Generic, Protocol, TypeVar, runtime_checkable

from anyio import CancelScope
from opentelemetry.trace import Tracer

from hfendpoint.core import Handler


I = TypeVar("I")
O = TypeVar("O")


@runtime_checkable
class Endpoint(Generic[I, O], Protocol):

    def __call__(self, request: I, tracer: Tracer, is_cancelled: CancelScope) -> O:
        ...

    @property
    def handler(self) -> Handler:
        ...