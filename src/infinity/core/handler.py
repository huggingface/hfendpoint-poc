from typing import Protocol, ClassVar


class Handler(Protocol):

    INPUT_TYPE: ClassVar
    OUTPUT_TYPE: ClassVar

    @property
    def router(self):
        ...