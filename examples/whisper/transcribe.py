import asyncio
import os.path
from contextlib import asynccontextmanager
from functools import partial
from io import BytesIO
from os import PathLike
from typing import Union, List, Tuple, Generator

import librosa
import numpy as np
from anyio import CancelScope
from asgi_correlation_id import CorrelationIdMiddleware, correlation_id
from fastapi import FastAPI, Response, status
from opentelemetry.trace import Tracer
from transformers import PreTrainedTokenizer
from vllm import SamplingParams, RequestOutput, CompletionOutput

from hfendpoint.audio import chunk_audio_with_duration
from hfendpoint.core import AsyncEndpoint
from hfendpoint.engines.vllm import VllmEngine, VllmGenerateParams
from hfendpoint.openai.audio.transcriptions import ApiRequest, ResponseFormat, TranscriptionHandler, Transcription, \
    VerboseTranscription, Segment
from librosa import load as load_audio_content
from vllm.engine.async_llm_engine import AsyncEngineArgs

from hfendpoint.openai.utils import compression_ratio

WHISPER_SEGMENT_DURATION_SEC = 30
WHISPER_SAMPLING_RATE = 22050


load_mono_audio_at_22050 = partial(load_audio_content, sr=22050, mono=True)


def process_chunk(tokenizer: PreTrainedTokenizer, ids: np.ndarray, request: ApiRequest) -> Generator:
    # Some constants
    k_timestamp_token = tokenizer.convert_tokens_to_ids("<|0.00|>")

    # Detect start of transcript token
    # sot_mask = ids == k_sot_token

    # Timestamps are expected to have ids greater than token_id(<|0.00|>)
    # We create a mask for all the potential tokens which are >= token_id(<|0.00|>)
    timestamps_mask = ids >= k_timestamp_token

    if np.any(timestamps_mask):
        # If we have a timestamp token, we need to check whether it's a final token or a final then the next
        is_single_ending_timestamp = np.array_equal(timestamps_mask[-2:], [False, True])

        # Iterate over timestamps
        timestamp_start, timestamp_end = 0.0, 0.0
        slice_start = 0

        for (t, position) in enumerate(np.flatnonzero(timestamps_mask)):
            timestamp = float(tokenizer.convert_ids_to_tokens([ids[position]])[0][2:-2])

            if t % 2 == 0:
                timestamp_end = timestamp

                # Retrieve segment info
                segment_ids = ids[slice_start: position]
                segment_text = tokenizer.decode(segment_ids)

                # Compute the avg_logprob and no_speech_probs for the segment
                # segment_logprobs = generation.logprobs[slice_start: position]
                # segment_probs_at_sot = np.exp([logprobs[nospeech_token_base] for logprobs in segment_logprobs])
                # segment_logprobs = [segment_logprobs[token] for token in segment_ids]

                # Materialize the segment in memory
                segment = Segment(
                    id=t,
                    seek=0,
                    start=timestamp_start,
                    end=timestamp_end,
                    text=segment_text,
                    tokens=segment_ids.tolist(),
                    temperature=request.temperature,
                    avg_logprob=None,
                    compression_ratio=compression_ratio(segment_text),
                    no_speech_prob=None
                    # no_speech_prob=segment_probs_at_sot / np.sum(segment_probs_at_sot)
                )

                yield segment, is_single_ending_timestamp

                # Update the start position
                slice_start = position
            else:
                timestamp_start = timestamp


def process_chunks(tokenizer: PreTrainedTokenizer, chunks: List[RequestOutput], request: ApiRequest) -> Tuple[List[Segment], str]:
    # k_nospeech_token = tokenizer.convert_tokens_to_ids("<|nospeech|>")
    # k_sot_token = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    materialized_segments, materialized_segments_tokens_acc = [], []

    # Iterate over segments
    for (idx, chunk) in enumerate(chunks):
        time_offset = idx * WHISPER_SEGMENT_DURATION_SEC
        segment_offset = len(materialized_segments)

        generation: CompletionOutput = chunk.outputs[-1]
        ids: np.ndarray = np.asarray(generation.token_ids)

        for (segment, _is_continuation) in process_chunk(tokenizer, ids, request):
            segment.id += segment_offset
            segment.start += time_offset
            segment.end += time_offset

            materialized_segments.append(segment)
            print(len(materialized_segments))

        # Accumulate the tokens for full decoding
        materialized_segments_tokens_acc += generation.token_ids

    text = tokenizer.decode(
        materialized_segments_tokens_acc,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return materialized_segments, text


def create_prompt(audio: np.ndarray, sampling_rate: int, timestamp_marker: int, request: ApiRequest):
    # TODO: We assume english for now
    k_english_token = 50259
    k_timestamp_marker = f"<|{timestamp_marker if request.response_format == ResponseFormat.VERBOSE_JSON else 0:.2f}|>"
    k_timestamp_marker_token = 50365

    return {
        "encoder_prompt": {
            "prompt": "",
            "multi_modal_data": {
                "audio": (audio, sampling_rate)
            }
        },
        "decoder_prompt": {
            # <|startoftranscript|><|{request.language}|><|transcribe|>{timestamp_marker}
            "prompt_token_ids": [50258, k_english_token, 50360, k_timestamp_marker_token]
        }
    }


class WhisperEndpoint(AsyncEndpoint[ApiRequest, Union[Transcription, VerboseTranscription]]):

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

    async def _handle_inference_stream(self, params: VllmGenerateParams, is_cancelled: CancelScope):
        async for step in self._engine.schedule(params):
            if is_cancelled.cancel_called:
                await self._engine.cancel(step.request_id)

        return step

    async def __call__(self, request: ApiRequest, tracer: Tracer, is_cancelled: CancelScope) -> Union[Response, Transcription, VerboseTranscription]:
        if len(request.file):
            with tracer.start_as_current_span("whisper.audio.demux"):
                audio, sampling = await self.run_in_executor(load_mono_audio_at_22050, BytesIO(request.file))

            # Handle parameters
            x_request_id = correlation_id.get()
            is_verbose_output = request.response_format == ResponseFormat.VERBOSE_JSON

            # Start inference
            chunks_handle = []
            with tracer.start_as_current_span("whisper.infer"):
                # Chunk audio in pieces
                audio_chunks = chunk_audio_with_duration(
                    audio, maximum_duration_sec=WHISPER_SEGMENT_DURATION_SEC, sampling_rate=WHISPER_SAMPLING_RATE
                )

                # Retrieve model config for caching
                tokenizer = await self._engine.tokenizer()
                model_config = await self._engine.config()

                # Submit audio pieces to the batcher and gather them all
                for (audio_chunk_id, audio_chunk) in enumerate(audio_chunks):
                    # Compute initial prompt for the segment
                    timestamp = audio_chunk_id * WHISPER_SEGMENT_DURATION_SEC
                    prompt = create_prompt(tokenizer, audio_chunk, sampling, timestamp, request)

                    # Submit for inference on the segment
                    params = VllmGenerateParams(
                        prompt=prompt,
                        sampling_params=SamplingParams.from_optional(
                            # output_kind=RequestOutputKind.FINAL_ONLY,  # Change if streaming
                            max_tokens=model_config.max_target_positions - 4,
                            skip_special_tokens=False,
                            detokenize=False,
                            temperature=request.temperature,
                            logprobs=is_verbose_output,

                        ),
                        request_id=f"{x_request_id}-{audio_chunk_id}"
                    )

                    # Keep track of the background task
                    chunks_handle += [self._handle_inference_stream(params, is_cancelled)]

                # Wait for all the segment to complete
                text_chunks = await asyncio.gather(*chunks_handle)

            if not is_cancelled.cancel_called:
                with tracer.start_as_current_span("whisper.post_process"):

                    segments, text = await self.run_in_executor(process_chunks, tokenizer, text_chunks, request)

                    if is_verbose_output:
                        return VerboseTranscription(
                            text=text,
                            duration=librosa.get_duration(y=audio, sr=sampling),
                            language="en",
                            segments=segments,
                            word=None
                        )
                    else:
                        return Transcription(text=text)

        return Response(status_code=status.HTTP_204_NO_CONTENT)


# Move below to a wrapper CLI call? Sounds pretty generic someway
@asynccontextmanager
async def endpoint(app: FastAPI) -> WhisperEndpoint:
    from os import environ as envvar

    if (model_id := envvar.get("MODEL_ID")) is None:
        raise ValueError("Unable to determine model to load")
    else:
        model_id = os.path.expanduser(model_id)

    print(f"Loading model {model_id}")

    # Create the target endpoint
    instance = WhisperEndpoint(model_id)
    app.include_router(instance.handler.router, tags=["openai", "transcriptions"])
    # app.include_router(instance., tags=["hfendpoint", "hpa"])

    yield

    del instance


# Allocate HTTP server
app = FastAPI(lifespan=endpoint)
app.add_middleware(CorrelationIdMiddleware)
