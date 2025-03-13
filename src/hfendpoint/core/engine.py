from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any

P = TypeVar('P', infer_variance=True)


class Engine(Generic[P], ABC):

    @abstractmethod
    async def schedule(self, params: P):
        raise NotImplementedError("Engine::schedule is abstract")


    async def cancel(self, request_id: Any):
        raise NotImplementedError("Engine::cancel is abstract")