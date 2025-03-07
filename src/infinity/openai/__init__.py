from typing import Any
from weakref import WeakValueDictionary


__services__ = WeakValueDictionary()


def register_service(name: str, service: Any):
    if name in __services__:
        raise KeyError(f"Service {name} already exists: {__services__[name]}")

    __services__[name] = service


def delete_service(name: str):
    if name in __services__:
        del __services__[name]

def get_service(name: str) -> Any:
    return __services__[name]

