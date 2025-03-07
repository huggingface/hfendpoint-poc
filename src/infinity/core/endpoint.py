from typing import Generic, Protocol, TypeVar, runtime_checkable

from infinity.core import Engine, Handler


I = TypeVar("I")
O = TypeVar("O")


@runtime_checkable
class Endpoint(Generic[I, O], Protocol):

    def on_request(self, request: I) -> O:
        ...

    @property
    def handler(self) -> Handler:
        ...