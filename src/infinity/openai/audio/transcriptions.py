import weakref
from abc import ABC
from dataclasses import dataclass
from enum import Enum

from fastapi.exceptions import RequestValidationError
from opentelemetry import trace
from typing import Annotated, List, Union

from fastapi import APIRouter, Depends, File, Response, status, HTTPException
from fastapi.openapi.utils import get_openapi

from infinity import Handler
from infinity.openai import get_service, register_service


ENDPOINT_NAME = "endpoint"

ISO639_1_SUPPORTED_LANGS = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "zh": "Chinese",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "is": "Icelandic",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "ms": "Malay",
    "mr": "Marathi",
    "mi": "Maori",
    "ne": "Nepali",
    "no": "Norwegian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "cy": "Welsh"
}


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


class TranscriptionResponseFormat(Enum):
    JSON = "json"
    SRT = "srt"
    TEXT = "text"
    VERBOSE_JSON = "verbose_json"
    VTT = "vtt"


@dataclass
class TranscriptionRequest:
    file: Annotated[bytes, File()]
    model: str | None = None
    language: str = "en"
    prompt: str | None = None
    temperature: float = 0.0
    response_format: TranscriptionResponseFormat = TranscriptionResponseFormat.JSON

router = APIRouter()


@router.post("/v1/audio/transcriptions")
async def transcription(
    request: Annotated[Depends(TranscriptionRequest), Depends(TranscriptionRequest)],
    response: Response
) -> Union[Transcription, VerboseTranscription]:
    tracer = trace.get_tracer("huggingface.endpoints.audio.transcriptions")

    if request.language not in ISO639_1_SUPPORTED_LANGS:
        raise HTTPException(400, f"{request.language} is not a valid ISO-639-1 language format.")

    with tracer.start_as_current_span("on_request") as span:
        try:
            service = get_service(ENDPOINT_NAME)
            return await service(request, tracer)
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



