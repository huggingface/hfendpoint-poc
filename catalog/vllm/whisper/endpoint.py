import asyncio
import zlib
from functools import lru_cache
from io import BytesIO
from typing import Sequence, Any, List, Tuple, Generator, Optional

import numpy as np
import torch
from hfendpoints.openai import Context, run
from hfendpoints.openai.audio import (
    AutomaticSpeechRecognitionEndpoint,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseKind,
    SegmentBuilder,
    Segment,
    Transcription,
    VerboseTranscription,
)
from librosa import load as load_audio, get_duration
from loguru import logger
from transformers import PreTrainedTokenizer
from vllm import (
    AsyncEngineArgs,
    AsyncLLMEngine,
    CompletionOutput,
    RequestOutput,
    SamplingParams,
    TokensPrompt,
)
from vllm.sequence import SampleLogprobs

from hfendpoints import Handler


def chunk_audio_with_duration(
        audio: np.ndarray, maximum_duration_sec: int, sampling_rate: int
) -> Sequence[np.ndarray]:
    """
    Chunk a mono audio timeseries so that each chunk is as long as `maximum_duration_sec`.
    Chunks are evenly distributed except the last one which might be shorter
    :param audio: The mono timeseries waveform of the audio
    :param maximum_duration_sec: The maximum length, in seconds, for each chunk
    :param sampling_rate: The number of samples to represent one second of audio
    :return: List of numpy array representing the chunk
    """

    # We pad the input so that every chunk length is `max_duration_sec`
    max_duration_samples = sampling_rate * maximum_duration_sec
    padding = max_duration_samples - np.remainder(len(audio), max_duration_samples)
    audio = np.pad(audio, (0, padding), constant_values=0.0)
    return np.split(audio, len(audio) // max_duration_samples)


def compression_ratio(text: str) -> float:
    """

    :param text:
    :return:
    """
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def create_prompt(
        audio: np.ndarray,
        sampling_rate: int,
        timestamp_marker: int,
        is_verbose_response: bool,
):
    """

    :param audio:
    :param sampling_rate:
    :param timestamp_marker:
    :param is_verbose_response:
    :return:
    """
    # TODO: We assume english for now
    k_english_token = 50259
    k_timestamp_marker = f"<|{timestamp_marker if is_verbose_response else 0:.2f}|>"
    k_timestamp_marker_token = 50365

    return {
        "encoder_prompt": {
            "prompt": "",
            "multi_modal_data": {"audio": (audio, sampling_rate)},
        },
        "decoder_prompt": {
            # <|startoftranscript|><|{request.language}|><|transcribe|>{timestamp_marker}
            "prompt_token_ids": [
                50258,
                k_english_token,
                50360,
                k_timestamp_marker_token,
            ]
        },
    }


def create_params(
        max_tokens: int, temperature: float, is_verbose: bool
) -> SamplingParams:
    """

    :param max_tokens:
    :param temperature:
    :param is_verbose:
    :return:
    """
    return SamplingParams.from_optional(
        # output_kind=RequestOutputKind.FINAL_ONLY,  # Change if streaming
        max_tokens=max_tokens,
        skip_special_tokens=False,
        detokenize=False,
        temperature=temperature,
        logprobs=100 if is_verbose else None,
    )


def get_avg_logprob(logprobs: SampleLogprobs) -> float:
    """

    :param logprobs:
    :return:
    """
    return sum(next(iter(_step_.values())).logprob for _step_ in logprobs) / float(
        len(logprobs)
    )


def process_chunk(
        tokenizer: PreTrainedTokenizer,
        ids: np.ndarray,
        logprobs: torch.Tensor,
        request: TranscriptionRequest,
        segment_offset: int,
        timestamp_offset: int,
) -> Generator:
    # Some constants
    k_timestamp_token = lru_cache(tokenizer.convert_tokens_to_ids)("<|0.00|>")

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

        for t, position in enumerate(np.flatnonzero(timestamps_mask)):
            timestamp = float(tokenizer.convert_ids_to_tokens([ids[position]])[0][2:-2])

            if t % 2 == 0:
                timestamp_end = timestamp

                # Retrieve segment info
                segment_ids = ids[slice_start:position]
                segment_text = tokenizer.decode(segment_ids)

                # Compute the avg_logprob
                avg_logprob = get_avg_logprob(logprobs)

                # no-speech logprob
                # no_speech_token_id = lru_cache(tokenizer.convert_tokens_to_ids("<|nospeech|>"))
                # no_speech_logprob = logprobs[no_speech_token_id]

                # Materialize the segment in memory
                segment = (
                    SegmentBuilder()
                    .id(segment_offset + t)
                    .start(timestamp_offset + timestamp_start)
                    .end(timestamp_offset + timestamp_end)
                    .text(segment_text)
                    .tokens(segment_ids.tolist())
                    .temperature(request.temperature)
                    .avg_logprob(avg_logprob)
                    .compression_ratio(compression_ratio(segment_text))
                    .build()
                )

                yield segment, is_single_ending_timestamp

                # Update the start position
                slice_start = position
            else:
                timestamp_start = timestamp


def process_chunks(
        tokenizer: PreTrainedTokenizer,
        chunks: List[RequestOutput],
        request: TranscriptionRequest,
) -> Tuple[List[Segment], str]:
    # k_nospeech_token = tokenizer.convert_tokens_to_ids("<|nospeech|>")
    # k_sot_token = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    materialized_segments, materialized_segments_tokens_acc = [], []

    # Iterate over segments
    for idx, chunk in enumerate(chunks):
        time_offset = idx * WhisperHandler.WHISPER_SEGMENT_DURATION_SEC
        segment_offset = len(materialized_segments)

        generation: CompletionOutput = chunk.outputs[-1]
        ids: np.ndarray = np.asarray(generation.token_ids)
        logprobs = generation.logprobs

        for segment, _is_continuation in process_chunk(
                tokenizer, ids, logprobs, request, segment_offset, time_offset
        ):
            materialized_segments.append(segment)

        # Accumulate the tokens for full decoding
        materialized_segments_tokens_acc += generation.token_ids

    text = tokenizer.decode(
        materialized_segments_tokens_acc,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return materialized_segments, text


class WhisperHandler(Handler[TranscriptionRequest, TranscriptionResponse]):
    WHISPER_SEGMENT_DURATION_SEC = 30
    WHISPER_SAMPLING_RATE = 22050

    __slots__ = ("_engine",)

    def __init__(self, model_id_or_path: str):
        super().__init__(model_id_or_path)

        self._engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model_id_or_path,
                task="transcription",
                device="auto",
                dtype="bfloat16",
                kv_cache_dtype="fp8",
                enforce_eager=False,
                enable_prefix_caching=True,
                max_logprobs=100,  # TODO(mfuntowicz) : Set from config?
            )
        )

    async def __call__(
            self, request: TranscriptionRequest, ctx: Context
    ) -> TranscriptionResponse:
        with logger.contextualize(request_id=ctx.request_id):
            with memoryview(request) as audio:

                # Check if we need to enable the verbose path
                is_verbose = (
                        request.response_kind == TranscriptionResponseKind.VERBOSE_JSON
                )

                tokenizer = asyncio.create_task(self._engine.get_tokenizer())
                model_config = asyncio.create_task(self._engine.get_model_config())

                # Decode audio from librosa (for now)
                # TODO: Use native (Rust provided) decoding
                (waveform, sampling) = load_audio(BytesIO(audio), sr=22050, mono=True)
                logger.debug(
                    f"Successfully decoded {len(waveform)} bytes PCM audio chunk"
                )

                max_tokens = (await model_config).max_model_len - 4

                # Chunk audio in pieces
                audio_chunks = chunk_audio_with_duration(
                    waveform,
                    maximum_duration_sec=WhisperHandler.WHISPER_SEGMENT_DURATION_SEC,
                    sampling_rate=WhisperHandler.WHISPER_SAMPLING_RATE,
                )

                # Submit audio pieces to the batcher and gather them all
                chunks_handle = []
                for audio_chunk_id, audio_chunk in enumerate(audio_chunks):
                    # Generate suffixed request-id to submit and identify through vLLM scheduler
                    request_id = f"{ctx.request_id}-{audio_chunk_id}"

                    timestamp = (
                            audio_chunk_id * WhisperHandler.WHISPER_SEGMENT_DURATION_SEC
                    )

                    # Compute initial prompt for the segment
                    prompt = create_prompt(audio_chunk, sampling, timestamp, request)
                    params = create_params(max_tokens, request.temperature, is_verbose)

                    # Submit for inference on the segment & keep track of the background task
                    chunks_handle += [
                        self._handle_inference_stream(
                            prompt, params, request_id, cancel=None
                        )
                    ]

                    # Wait for all the segment to complete
                text_chunks = await asyncio.gather(*chunks_handle)

                # if not is_cancelled.cancel_called:
                tokenizer = await tokenizer
                segments, text = await asyncio.get_event_loop().run_in_executor(
                    None, process_chunks, tokenizer, text_chunks, request
                )

                match request.response_kind:
                    case TranscriptionResponseKind.VERBOSE_JSON:
                        return TranscriptionResponse.verbose(
                            VerboseTranscription(
                                text=text,
                                duration=get_duration(y=waveform, sr=sampling),
                                language="en",
                                segments=segments,
                                # word=None
                            )
                        )
                    case TranscriptionResponseKind.JSON:
                        return TranscriptionResponse.json(text)

                    case TranscriptionResponseKind.TEXT:
                        return TranscriptionResponse.text(text)

    async def _handle_inference_stream(
            self,
            prompt: TokensPrompt,
            params: SamplingParams,
            request_id: str,
            cancel: Optional[Any] = None,
    ):
        async for step in self._engine.generate(prompt, params, request_id):
            # if is_cancelled.cancel_called:
            #     await self._engine.cancel(step.request_id)
            pass

        return step


def entrypoint():
    endpoint = AutomaticSpeechRecognitionEndpoint(
        WhisperHandler("openai/whisper-large-v3")
    )
    run(endpoint, "0.0.0.0", 8000)


if __name__ == "__main__":
    entrypoint()
