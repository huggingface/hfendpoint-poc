from abc import ABC
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi.responses import JSONResponse
from opentelemetry import trace
from typing import Annotated, List, Union

from fastapi import APIRouter, File, Form, Response, Request, HTTPException
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel

from infinity import Handler
from infinity.openai import get_service, register_service, scoped_cancellation_handler

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


class Transcription(BaseModel):
    """
    Represents a transcription response returned by model, based on the provided input.
    """

    text: str


class VerboseTranscription(Transcription):
    """
    Represents a transcription response returned by model, based on the provided input.
    """

    language: str
    duration: int
    word: List[Word]


class ResponseFormat(str, Enum):
    JSON = "json"
    SRT = "srt"
    TEXT = "text"
    VERBOSE_JSON = "verbose_json"
    VTT = "vtt"



class ApiRequest(BaseModel):
    file: Annotated[bytes, File()]
    model: str | None = None
    language: str = "en"
    prompt: str | None = None
    temperature: float = 0.0
    response_format: ResponseFormat = ResponseFormat.JSON

router = APIRouter()


@router.post("/v1/audio/transcriptions", response_model=Union[Transcription, VerboseTranscription])
async def transcription(
    params: Annotated[ApiRequest, Form()],
    request: Request,
) -> Response:
        tracer = trace.get_tracer("huggingface.endpoints.audio.transcriptions")

        if params.language not in ISO639_1_SUPPORTED_LANGS:
            raise HTTPException(400, f"{params.language} is not a valid ISO-639-1 language format.")

        with tracer.start_as_current_span("on_request") as span:
            try:
                service = get_service(ENDPOINT_NAME)

                async with scoped_cancellation_handler(request) as has_disconnected:
                    # TODO: Better handle failure in response - Rust's Result<T, E> ?
                    result: Union[Transcription, VerboseTranscription] = await service(params, tracer, has_disconnected)
                    return JSONResponse(result.model_dump())

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

    INPUT_TYPE = ApiRequest
    OUTPUT_TYPE = Union[Transcription, VerboseTranscription]

    def __init__(self, endpoint: "Endpoint"):
        register_service(ENDPOINT_NAME, endpoint)  # Register will make a weakref to $endpoint

    @property
    def router(self):
        return router



