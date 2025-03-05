from abc import ABC, abstractmethod


class Handler(ABC):
    @abstractmethod
    @property
    def router(self):
        raise NotImplementedError("Handler::router is abstract")