from abc import ABC
from dataclasses import dataclass
from io import BytesIO
from opentelemetry import trace
from typing import Annotated, List, Optional, Union

from fastapi import APIRouter, Depends, File
from fastapi.openapi.utils import get_openapi
from librosa import load as load_audio_content

from infinity import Handler, Endpoint
from infinity.engines.vllm import VllmEngine
from infinity.openai.utils import raise_


@dataclass
class Word:
    word: str
    start : int
    end: int


@dataclass
class Transcription:
    """
    Represents a transcription response returned by model, based on the provided input.
    """

    text: str


@dataclass
class VerboseTranscription(Transcription):
    """
    Represents a transcription response returned by model, based on the provided input.
    """

    language: str
    duration: int
    word: List[Word]


router = APIRouter()

@router.post("/v1/audio/transcriptions")
async def transcription(
        service: Annotated[Endpoint, Depends(Endpoint)],
        file: Annotated[bytes, File()],
        model: Optional[str] = None,
        language: Optional[str] = "en",
        prompt: Optional[str] = None,
        temperature: Optional[float] = 0.0,
) -> Union[Transcription, VerboseTranscription]:
    tracer =  trace.get_tracer("huggingface.endpoints.audio")
    with tracer.start_as_current_span("transcriptions"):

        engine: VllmEngine = service.engine

        # Unmarshall the audio content
        with tracer.start_as_current_span("audio"):
            audio, sampling_rate = load_audio_content(BytesIO(file))

        with tracer.start_as_current_span("on_request"):
            pass


    return Transcription("World")

def openapi_transcriptions():
    if router.openapi_schema:
        return router.openapi_schema

    router.openapi_schema = get_openapi(
        title="Infinity Audio Transcriptions Endpoint",
        version="1.0.0",
        summary="OpenAI API compatible Audio Transcriptions",
        description="",
        routes=router.routes,
    )
    return router.openapi_schema

router.openapi = openapi_transcriptions


class TranscriptionHandler(Handler, ABC):
    def router(self):
        return router



