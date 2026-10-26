from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "0+unknown"


def runtime_identity() -> dict[str, str]:
    return {
        "engine": "phonon",
        "phononsynth": _version("phononsynth"),
        "onnxvoice": _version("onnxvoice"),
        "tokenizers": _version("tokenizers"),
        "request_api": "1",
    }
