from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any

from transformers import PreTrainedTokenizer, PretrainedConfig

P = TypeVar('P', infer_variance=True)


class Engine(Generic[P], ABC):

    @abstractmethod
    async def tokenizer(self) -> PreTrainedTokenizer:
        raise NotImplementedError("Engine::tokenizer is abstract")

    @abstractmethod
    async def config(self) -> PretrainedConfig:
        raise NotImplementedError("Engine::config is abstract")


    @abstractmethod
    async def schedule(self, params: P):
        raise NotImplementedError("Engine::schedule is abstract")


    @abstractmethod
    async def cancel(self, request_id: Any):
        raise NotImplementedError("Engine::cancel is abstract")