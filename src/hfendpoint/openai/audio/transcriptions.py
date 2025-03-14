from abc import ABC
from enum import Enum

from fastapi.responses import JSONResponse
from opentelemetry import trace
from typing import Annotated, List, Union, Optional

from fastapi import APIRouter, File, Form, Response, Request, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

from hfendpoint import Handler
from hfendpoint.openai import get_service, register_service, scoped_cancellation_handler
from pydantic.types import NonNegativeInt, NonNegativeFloat

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

class Segment(BaseModel):
    # Unique identifier of the segment.
    id: NonNegativeInt

    # Seek offset of the segment.
    seek: NonNegativeInt

    # Start time of the segment in seconds.
    start: NonNegativeFloat = Field(decimal_places=2)

    # End time of the segment in seconds.
    end: NonNegativeFloat = Field(decimal_places=2)

    # Text content of the segment.
    text: str

    # Array of token IDs for the text content.
    tokens: List[NonNegativeInt]

    # Temperature parameter used for generating the segment.
    temperature: float

    # Average logprob of the segment. If the value is lower than -1, consider the logprobs failed.
    avg_logprob: float = Field(default=0.0, decimal_places=2)

    # Compression ratio of the segment. If the value is greater than 2.4, consider the compression failed.
    compression_ratio: NonNegativeFloat = Field(default=0.0, decimal_places=2)

    # Probability of no speech in the segment. If the value is higher than 1.0 and the avg_logprob is below -1, consider this segment silent.
    no_speech_prob: NonNegativeFloat = Field(default=0.0, decimal_places=2)


class Word(BaseModel):
    word: str
    start : NonNegativeInt
    end: NonNegativeInt


class Transcription(BaseModel):
    """
    Represents a transcription response returned by model, based on the provided input.
    """

    text: str


class VerboseTranscription(Transcription):
    """
    Represents a transcription response returned by model, based on the provided input.
    """

    # The language of the input audio.
    language: str

    # The duration of the input audio.
    duration: NonNegativeFloat

    # Segments of the transcribed text and their corresponding details.
    segments: List[Segment]

    # Extracted words and their corresponding timestamps. Not supported yet.
    word: Optional[List[Word]]

    # The transcribed text.
    text: str


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
                print(e)
                raise e


@router.get("/health")
async def health() -> PlainTextResponse:
    return PlainTextResponse("Ok")


def openapi_transcriptions():
    if router.openapi_schema:
        return router.openapi_schema

    router.openapi_schema = get_openapi(
        title="Hugging Face Audio Transcriptions Endpoint",
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



