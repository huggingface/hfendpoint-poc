from abc import ABC
from typing import TYPE_CHECKING

from infinity import Handler

if TYPE_CHECKING:
    from fastapi import FastAPI


class OpenAiHandler(Handler, ABC):
    def register_routes(self, app: "FastAPI"):
        raise NotImplementedError("OpenAiHandler::register_routes is abstract")