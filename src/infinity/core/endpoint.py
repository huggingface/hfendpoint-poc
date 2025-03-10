from typing import Generic, Protocol, TypeVar, runtime_checkable

from opentelemetry.trace import Tracer

from infinity.core import Engine, Handler


I = TypeVar("I")
O = TypeVar("O")


@runtime_checkable
class Endpoint(Generic[I, O], Protocol):

    def __call__(self, request: I, tracer: Tracer) -> O:
        ...

    @property
    def handler(self) -> Handler:
        ...