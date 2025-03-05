from abc import ABC, abstractmethod

from infinity import Engine

class Endpoint(ABC):
    @abstractmethod
    @property
    def engine(self) -> Engine:
        raise NotImplementedError("Endpoint::engine is abstract")