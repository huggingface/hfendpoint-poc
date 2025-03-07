from contextlib import asynccontextmanager
from io import BytesIO
from os import PathLike
from typing import Union

from fastapi import FastAPI
from vllm import SamplingParams

from infinity import Endpoint
from infinity.engines.vllm import VllmEngine, VllmGenerateParams
from infinity.openai.audio.transcriptions import TranscriptionHandler, TranscriptionRequest, Transcription, VerboseTranscription
from librosa import load as load_audio_content
from vllm.engine.async_llm_engine import AsyncEngineArgs


class WhisperEndpoint(Endpoint[TranscriptionHandler.INPUT_TYPE, TranscriptionHandler.OUTPUT_TYPE]):

    __slots__ = ("_engine", "_handler")

    def __init__(self, model: Union[str, PathLike]):
        super().__init__()

        self._engine = VllmEngine(AsyncEngineArgs(
            model,
            device="auto",
            enforce_eager=True,
            kv_cache_dtype="fp8"
        ))

        self._handler = TranscriptionHandler(self)

    @property
    def handler(self):
        return self._handler

    async def on_request(self, request: TranscriptionRequest) -> Union[Transcription, VerboseTranscription]:
        if len(request.file):
            audio, sampling = load_audio_content(BytesIO(request.file), sr=22050)

            output = await self._engine.schedule(VllmGenerateParams(
                prompt={
                    "prompt": "<|startoftranscript|>",
                    "multi_modal_data": {
                        "audio": (audio, sampling)
                    }
                },
                sampling_params=SamplingParams(temperature=request.temperature, max_tokens=1024),
                request_id=""
            ))

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


