from abc import ABC, abstractmethod
from typing import Generic, TypeVar

P = TypeVar('P', infer_variance=True)


class Engine(Generic[P], ABC):

    @abstractmethod
    async def schedule(self, params: P):
        raise NotImplementedError("Engine::schedule is abstract")