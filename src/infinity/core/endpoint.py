from typing import Protocol

from infinity.core import Engine, Handler

class Endpoint(Protocol):
    @property
    def engine(self) -> Engine:
        ...

    @property
    def handler(self) -> Handler:
        ...