import weakref
from abc import ABC
from dataclasses import dataclass
from opentelemetry import trace
from typing import Annotated, List, Union

from fastapi import APIRouter, Depends, File
from fastapi.openapi.utils import get_openapi

from infinity import Handler
from infinity.openai import get_service, register_service


ENDPOINT_NAME = "endpoint"


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


@dataclass
class TranscriptionRequest:
    file: Annotated[bytes, File()]
    model: str | None = None
    language: str = "en"
    prompt: str | None = None
    temperature: float = 0.0

router = APIRouter()

@router.post("/v1/audio/transcriptions")
async def transcription(
    request: Annotated[Depends(TranscriptionRequest), Depends(TranscriptionRequest)]
) -> Union[Transcription, VerboseTranscription]:
    tracer = trace.get_tracer("huggingface.endpoints.audio.transcriptions")
    with tracer.start_as_current_span("on_request") as span:
        try:
            return await get_service(ENDPOINT_NAME).on_request(request)
        except Exception as e:
            span.add_event(e)
            raise e


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

    INPUT_TYPE = TranscriptionRequest
    OUTPUT_TYPE = Union[Transcription, VerboseTranscription]


    def __init__(self, endpoint: "Endpoint"):
        register_service(ENDPOINT_NAME, endpoint)  # Register will make a weakref to $endpoint

    @property
    def router(self):
        return router



