from abc import ABC
from dataclasses import asdict, dataclass
from typing import TypedDict, Union, Any

from infinity import Engine
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams, TextPrompt, TokensPrompt
from vllm.usage.usage_lib import UsageContext


INFINITY_VLLM_USAGE_CONTEXT = "hf.endpoints.vllm"


@dataclass
class VllmGenerateParams:
    prompt: Union[str, TextPrompt, TokensPrompt, TypedDict]
    sampling_params: SamplingParams
    request_id: str


class VllmEngine(Engine[VllmGenerateParams], ABC):
    __slots__ = ("_engine", )

    def __init__(self, engine_or_args: Union["AsyncEngineArgs", "AsyncLLMEngine"]):
        if isinstance(engine_or_args, AsyncEngineArgs):
            engine = AsyncLLMEngine.from_engine_args(
                engine_or_args,
                usage_context=UsageContext.API_SERVER
            )
        else:
            engine = engine_or_args

        self._engine = engine

    async def schedule(self, params: VllmGenerateParams):
        async for step in self._engine.generate(**asdict(params)):
            # Handle cancellation
            print(f"intermediate step: {step}")
        return step





