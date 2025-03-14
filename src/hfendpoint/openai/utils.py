import zlib


def raise_(exception: Exception):
    raise exception


def compression_ratio(text: str) -> float:
        text_bytes = text.encode("utf-8")
        return len(text_bytes) / len(zlib.compress(text_bytes))