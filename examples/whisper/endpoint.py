from contextlib import asynccontextmanager
from io import BytesIO
from os import PathLike
from typing import Union

from fastapi import FastAPI
from opentelemetry.trace import Tracer
from vllm import SamplingParams

from infinity import Endpoint
from infinity.engines.vllm import VllmEngine, VllmGenerateParams
from infinity.openai.audio.transcriptions import TranscriptionHandler, TranscriptionRequest, Transcription, \
    VerboseTranscription, TranscriptionResponseFormat
from librosa import load as load_audio_content
from vllm.engine.async_llm_engine import AsyncEngineArgs


class WhisperEndpoint(Endpoint[TranscriptionHandler.INPUT_TYPE, TranscriptionHandler.OUTPUT_TYPE]):

    __slots__ = ("_engine", "_handler")

    def __init__(self, model: Union[str, PathLike]):
        super().__init__()

        self._engine = VllmEngine(AsyncEngineArgs(
            model,
            device="auto",
            enforce_eager=False,
            kv_cache_dtype="fp8"
        ))

        self._handler = TranscriptionHandler(self)

    @property
    def handler(self):
        return self._handler

    @staticmethod
    def with_timestamp_marker(is_verbose: bool) -> str:
        return f"<|{'0.00' if is_verbose else 'notimestamps'}|>"

    async def __call__(self, request: TranscriptionRequest, tracer: Tracer) -> Union[Transcription, VerboseTranscription]:
        if len(request.file):
            with tracer.start_as_current_span("whisper.audio.demux"):
                audio, sampling = load_audio_content(BytesIO(request.file), sr=22050)

            print(f"Response format: {request.response_format}")

            is_verbose_output = request.response_format == TranscriptionResponseFormat.VERBOSE_JSON
            timestamp_marker = WhisperEndpoint.with_timestamp_marker(is_verbose=is_verbose_output)
            with tracer.start_as_current_span("whisper.infer"):
                lang = "en"
                output = await self._engine.schedule(VllmGenerateParams(
                    prompt={
                        "encoder_prompt": {
                            "prompt": "",
                            "multi_modal_data": {
                                "audio": (audio, sampling)
                            }
                        },
                        "decoder_prompt": f"<|startoftranscript|><|{lang}|><|transcribe|>{timestamp_marker}"
                    },
                    sampling_params=SamplingParams(temperature=request.temperature, max_tokens=1024),
                    request_id=""
                ))

            with tracer.start_as_current_span("whisper.post_process"):
                tokenizer = await self._engine._engine.get_tokenizer()
                raw_decode = tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=False)
                print(raw_decode)
                return Transcription(output.outputs[0].text)

        return Transcription("")


@asynccontextmanager
async def endpoint(app: FastAPI) -> WhisperEndpoint:
    # Create the target endpoint
    instance = WhisperEndpoint("openai/whisper-large-v3")
    app.include_router(instance.handler.router, tags=["transcriptions"])

    yield

    del instance

# Allocate HTTP server
app = FastAPI(lifespan=endpoint)


