from abc import ABC
from typing import Any, Dict, Optional, Union

from vllm.v1.engine.llm_engine import EngineArgs, LLMEngine
from vllm.usage.usage_lib import UsageContext

from infinity.core import Handler


INFINITY_VLLM_USAGE_CONTEXT = "hf.endpoints.vllm"


class VllmEngine(Handler, ABC):

    __slots__ = ("_engine", )


    def __init__(self, engine_or_args: Union["EngineArgs", "LLMEngine"]):

        if isinstance(engine_or_args, EngineArgs):
            engine = LLMEngine.from_engine_args(engine_or_args, UsageContext.API_SERVER, None, True)
        else:
            engine = engine_or_args

        self._engine = engine


    async def schedule(self, params: Dict[str, Any]):
        self._engine.add_request()