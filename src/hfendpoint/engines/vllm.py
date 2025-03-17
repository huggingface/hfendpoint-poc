from abc import ABC
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import TypedDict, Union, Any

from transformers import PreTrainedTokenizer, PretrainedConfig
from typing_extensions import AsyncGenerator

from hfendpoint import Engine
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams, TextPrompt, TokensPrompt, RequestOutput
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

    async def config(self) -> PretrainedConfig:
        mconfig = await self._engine.get_model_config()
        return mconfig.hf_config

    async def tokenizer(self) -> PreTrainedTokenizer:
        return await self._engine.get_tokenizer()

    async def schedule(self, params: VllmGenerateParams) -> AsyncGenerator[RequestOutput]:
        async for step in self._engine.generate(**asdict(params)):
            yield step

    async def cancel(self, request_id: Any):
        await self._engine.abort(request_id)




