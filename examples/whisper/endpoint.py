from os import PathLike
from tokenize import endpats
from typing import Union

from fastapi import FastAPI
from infinity import Endpoint, Engine, Handler
from infinity.engines.vllm import VllmEngine
from infinity.openai.audio.transcriptions import TranscriptionHandler
from vllm.v1.engine.llm_engine import EngineArgs

class WhisperEndpoint(Endpoint):

    def __init__(self, model: Union[str, PathLike]):
        super().__init__()

        # self._engine = VllmEngine(
        #     EngineArgs(model, device="cpu")
        # )
        self._engine = None
        self._handler = TranscriptionHandler()

    @property
    def engine(self) -> Engine:
        return self._engine


    @property
    def handler(self) -> Handler:
        return self._handler


local_model_path = "/home/mfuntowicz/.cache/huggingface/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1"

# Create the target endpoint
endpoint = WhisperEndpoint(local_model_path)

# Allocate HTTP server
app = FastAPI()
app.include_router(endpoint.handler.router, tags=["transcriptions"])

