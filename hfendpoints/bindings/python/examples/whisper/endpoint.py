import logging

from hfendpoints.openai.audio import AutomaticSpeechRecognitionEndpoint, TranscriptionRequest, TranscriptionResponse, \
    Segment, Transcription, VerboseTranscription

from hfendpoints import Handler, run


class WhisperHandler(Handler[TranscriptionRequest, TranscriptionResponse]):
    __slots__ = ("_engine",)

    def __init__(self, model_id_or_path: str):
        super().__init__(model_id_or_path)

        # self._engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        #     model_id_or_path,
        #     task="transcription",
        #     device="auto",
        #     dtype="bfloat16",
        #     kv_cache_dtype="fp8",
        #     enforce_eager=True,
        #     enable_prefix_caching=True,
        # ))

    async def __call__(self, request: TranscriptionRequest, ctx) -> TranscriptionResponse:
        print(f"[Python] handler call: {request}")

        # with memoryview(request) as audio:
        #     (waveform, sr) = load_audio(BytesIO(audio), sr=22050, mono=True)

        return TranscriptionResponse.text("Awesome audio content")


def entrypoint():
    endpoint = AutomaticSpeechRecognitionEndpoint(WhisperHandler("openai/whisper-large-v3"))
    # endpoint.run("0.0.0.0", 8000)
    run(endpoint, "0.0.0.0", 8000)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    entrypoint()
