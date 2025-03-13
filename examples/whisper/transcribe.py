import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
from itertools import chain
from os import PathLike
from typing import Union, List, TypedDict

import numpy as np
from anyio import CancelScope
from asgi_correlation_id import CorrelationIdMiddleware, correlation_id
from fastapi import FastAPI
from opentelemetry.trace import Tracer
from transformers import PreTrainedTokenizer
from vllm import SamplingParams, RequestOutput, CompletionOutput
from vllm.sampling_params import RequestOutputKind

from hfendpoint import Endpoint
from hfendpoint.audio import chunk_audio_with_duration
from hfendpoint.engines.vllm import VllmEngine, VllmGenerateParams
from hfendpoint.openai.audio.transcriptions import ApiRequest, ResponseFormat, TranscriptionHandler, Transcription, \
    VerboseTranscription
from librosa import load as load_audio_content
from vllm.engine.async_llm_engine import AsyncEngineArgs



WHISPER_SEGMENT_DURATION_SEC = 30
WHISPER_SAMPLING_RATE = 22050


class WhisperEndpoint(Endpoint[ApiRequest, Union[Transcription, VerboseTranscription]]):

    __slots__ = ("_engine", "_handler")

    def __init__(self, model: Union[str, PathLike]):
        super().__init__()

        self._engine = VllmEngine(AsyncEngineArgs(
            model,
            device="auto",
            enforce_eager=True,
            dtype="bfloat16",
            kv_cache_dtype="fp8",
            enable_prefix_caching=True,
        ))

        self._handler = TranscriptionHandler(self)


    @property
    def handler(self):
        return self._handler

    @staticmethod
    def with_timestamp_marker(is_verbose: bool, current: int) -> str:
        return f"<|{f'{current}.00' if is_verbose else 'notimestamps'}|>"

    @staticmethod
    def create_prompt(audio, sampling_rate: int, timestamp_marker: str, request: ApiRequest) -> TypedDict:
        return {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": (audio, sampling_rate)
                }
            },
            "decoder_prompt": f"<|startoftranscript|><|{request.language}|><|transcribe|>{timestamp_marker}"
        }

    async def _postprocess(self, segments: List[RequestOutput]):
        tokenizer = await self._engine._engine.get_tokenizer()
        timestamp_token_base = tokenizer.convert_tokens_to_ids("<|0.00|>")

        for (idx, segment) in enumerate(segments):
            generation: CompletionOutput = segment.outputs[-1]
            ids: np.ndarray = np.asarray(generation.token_ids)

            print(f"{idx} -> {ids}")

            # Timestamps are expected to have ids greater than token_id(<|0.00|>)
            # We create a mask for all the potential tokens which are >= token_id(<|0.00|>)
            timestamps_mask = ids >= timestamp_token_base

            if np.any(timestamps_mask):
                tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
                text = tokenizer.convert_tokens_to_string(tokens)

                # If we have a timestamp token, we need to check whether it's a final token or a final then the next
                # is_single_ending_timestamp = np.array_equal(timestamps_mask[-2:], [False, True])

                print(text)


    async def _handle_inference_stream(self, params: VllmGenerateParams, is_cancelled: CancelScope):
        async for step in self._engine.schedule(params):
            if is_cancelled.cancel_called:
                await self._engine.cancel(step.request_id)

        return step

    async def __call__(self, request: ApiRequest, tracer: Tracer, is_cancelled: CancelScope) -> Union[Transcription, VerboseTranscription]:
        if len(request.file):
            with tracer.start_as_current_span("whisper.audio.demux"):
                audio, sampling = load_audio_content(BytesIO(request.file), sr=22050, mono=True)

            # Handle parameters
            x_request_id = correlation_id.get()
            is_verbose_output = request.response_format == ResponseFormat.VERBOSE_JSON

            # Start inference
            segments_handle = []
            with tracer.start_as_current_span("whisper.infer"):
                # Chunk audio in pieces
                segments = chunk_audio_with_duration(
                    audio, maximum_duration_sec=WHISPER_SEGMENT_DURATION_SEC, sampling_rate=WHISPER_SAMPLING_RATE
                )

                # Submit audio pieces to the batcher and gather them all
                for (segment_id, segment) in enumerate(segments):
                    # Compute current timestamp
                    timestamp_marker = WhisperEndpoint.with_timestamp_marker(
                        is_verbose=is_verbose_output,
                        current=segment_id * WHISPER_SEGMENT_DURATION_SEC
                    )

                    # Compute initial prompt for the segment
                    prompt = WhisperEndpoint.create_prompt(segment, sampling, timestamp_marker, request)

                    # Submit for inference on the segment
                    model_config = await self._engine._engine.get_model_config()
                    params = VllmGenerateParams(
                        prompt=prompt,
                        sampling_params=SamplingParams.from_optional(
                            # output_kind=RequestOutputKind.FINAL_ONLY,  # Change if streaming
                            max_tokens=model_config.max_model_len - 4,
                            skip_special_tokens=False,
                            detokenize=False,
                            temperature=request.temperature,
                        ),
                        request_id=f"{x_request_id}-{segment_id}"
                    )

                    segment_handle = self._handle_inference_stream(params, is_cancelled)
                    segments_handle.append(segment_handle)

                # Wait for all the segment to complete
                segments_output = await asyncio.gather(*segments_handle)

            if not is_cancelled.cancel_called:
                with tracer.start_as_current_span("whisper.post_process"):
                    await self._postprocess(segments_output)

        return Transcription(text="")


# Move below to a wrapper CLI call? Sounds pretty generic someway
@asynccontextmanager
async def endpoint(app: FastAPI) -> WhisperEndpoint:
    # Create the target endpoint
    instance = WhisperEndpoint("openai/whisper-large-v3")
    app.include_router(instance.handler.router, tags=["transcriptions"])

    yield

    del instance


# Allocate HTTP server
app = FastAPI(lifespan=endpoint)
app.add_middleware(CorrelationIdMiddleware)
