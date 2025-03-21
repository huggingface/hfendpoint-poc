from abc import ABC
from dataclasses import asdict, dataclass
from functools import partial
from typing import TypedDict, Union, Any, Optional, Tuple, List, Callable

from loguru import logger
from transformers import PreTrainedTokenizer, PretrainedConfig
from typing_extensions import AsyncGenerator
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams, TextPrompt, TokensPrompt, RequestOutput
from vllm.config import SchedulerConfig, CacheConfig, LoRAConfig
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.sequence import SequenceGroupMetadata
from vllm.usage.usage_lib import UsageContext

from hfendpoint import Engine
from hfendpoint.handlers.monitor import EngineMonitor, MonitoringStats

INFINITY_VLLM_USAGE_CONTEXT = "hf.endpoints.vllm"


class HpaObservableScheduler(Scheduler):

    __slots__ = ("_monitor", "_max_in_flight", )

    def __init__(
            self,
            scheduler_config: SchedulerConfig,
            cache_config: CacheConfig,
            lora_config: Optional[LoRAConfig],
            pipeline_parallel_size: int,
            output_proc_callback: Optional[Callable],
            monitor: EngineMonitor):
        super().__init__(scheduler_config, cache_config, lora_config, pipeline_parallel_size, output_proc_callback)

        self._monitor = monitor
        self._max_in_flight = scheduler_config.max_num_seqs

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, bool]:
        seq, outputs, allow_async_output_proc = super().schedule()

        self._monitor.broadcast_sync(MonitoringStats(
            in_flight=outputs.running_queue_size,
            in_queue=0,
            max_in_flight=self._max_in_flight,
        ))


        return seq, outputs, allow_async_output_proc


@dataclass
class VllmGenerateParams:
    prompt: Union[str, TextPrompt, TokensPrompt, TypedDict]
    sampling_params: SamplingParams
    request_id: str


class VllmEngine(Engine[VllmGenerateParams], ABC):
    __slots__ = ("_engine", )

    def __init__(
        self,
        engine_args: "AsyncEngineArgs",
        monitor: EngineMonitor
    ):
        engine_args.scheduler_cls = partial(HpaObservableScheduler, monitor=monitor)
        vllm_config = engine_args.create_engine_config(usage_context=UsageContext.API_SERVER)
        engine = AsyncLLMEngine.from_vllm_config(vllm_config)

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




