from abc import ABC, abstractmethod


class Engine(ABC):

    @abstractmethod
    async def schedule(self):
        raise NotImplementedError("Engine::schedule is abstract")