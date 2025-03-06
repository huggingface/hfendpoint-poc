from typing import Protocol


class Handler(Protocol):
    @property
    def router(self):
        ...