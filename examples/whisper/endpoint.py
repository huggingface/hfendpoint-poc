import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
from itertools import chain
from os import PathLike
from typing import Union

from anyio import CancelScope
from asgi_correlation_id import CorrelationIdMiddleware, correlation_id
from fastapi import FastAPI
from opentelemetry.trace import Tracer
from vllm import SamplingParams

from infinity import Endpoint
from infinity.audio import chunk_audio_with_duration
from infinity.engines.vllm import VllmEngine, VllmGenerateParams
from infinity.openai.audio.transcriptions import ApiRequest, ResponseFormat, TranscriptionHandler, Transcription, \
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
            enforce_eager=False,
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
            segments_output, join_handles = [], []
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

                    # Submit for inference on the segment
                    params = VllmGenerateParams(
                        prompt={
                            "encoder_prompt": {
                                "prompt": "",
                                "multi_modal_data": {
                                    "audio": (segment, sampling)
                                }
                            },
                            "decoder_prompt": f"<|startoftranscript|><|{request.language}|><|transcribe|>{timestamp_marker}"
                        },
                        sampling_params=SamplingParams(temperature=request.temperature, max_tokens=1024),
                        request_id=f"{x_request_id}-{segment_id}"
                    )

                    segment_handle = self._handle_inference_stream(params, is_cancelled)
                    segments_output.append(segment_handle)

                # Wait for all the segment to complete
                segments_output = await asyncio.gather(*segments_output)

            if not is_cancelled.cancel_called:
                with tracer.start_as_current_span("whisper.post_process"):
                    tokenizer = await self._engine._engine.get_tokenizer()
                    token_ids_all_segment = list(chain.from_iterable(map(lambda segment: segment.outputs[0].token_ids, segments_output)))

                    raw_decode = tokenizer.decode(token_ids_all_segment, skip_special_tokens=False, clean_up_tokenization_spaces=True)
                    return Transcription(text=raw_decode)

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
