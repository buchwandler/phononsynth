from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ManagedModelDependencyError, OnnxVoiceContractError, UnsupportedModelError

DEFAULT_CATALOG_URL = (
    "https://raw.githubusercontent.com/buchwandler/phonon-onnx-bundles/main/catalog/bundles.json"
)


@dataclass(frozen=True, slots=True)
class ResolvedPhononModel:
    ref: str
    model_id: str
    tokenizer_path: Path
    sample_rate: int
    voices: tuple[str, ...]
    default_voice: str | None
    metadata: dict[str, Any]
    installation: Any


def _module() -> Any:
    try:
        import onnxvoice
    except ModuleNotFoundError as exc:
        raise OnnxVoiceContractError("OnnxVoice is required for Phonon synthesis") from exc
    return onnxvoice


def normalize_ref(value: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError("model reference cannot be empty")
    return value if ":" in value else f"phonon:{value}"


def _manager(*, cache_dir: str | Path | None, offline: bool, catalog_url: str | None) -> Any:
    module = _module()
    source = catalog_url or os.getenv("PHONONSYNTH_CATALOG_URL") or DEFAULT_CATALOG_URL
    try:
        return module.OnnxVoice(
            cache_dir=cache_dir,
            catalog_sources={"phonon": source},
            offline=offline,
        )
    except TypeError as exc:
        raise OnnxVoiceContractError(
            "Installed OnnxVoice does not expose the expected manager constructor"
        ) from exc


def installation_to_model(installation: Any) -> ResolvedPhononModel:
    if str(getattr(installation, "system", "")) != "phonon":
        raise UnsupportedModelError("expected an OnnxVoice Phonon installation")
    try:
        tokenizer_path = Path(installation.artifact("tokenizer").path)
    except (KeyError, AttributeError) as exc:
        raise UnsupportedModelError("Phonon installation must contain tokenizer.json") from exc
    metadata = dict(getattr(installation, "metadata", {}) or {})
    sample_rate = int(
        getattr(installation, "sample_rate", None)
        or (metadata.get("runtime") or {}).get("sample_rate")
        or 24000
    )
    voices = tuple(str(value) for value in (getattr(installation, "voices", ()) or ()))
    default_voice = getattr(installation, "default_voice", None)
    return ResolvedPhononModel(
        ref=str(getattr(installation, "ref", f"phonon:{getattr(installation, 'id', 'unknown')}")),
        model_id=str(getattr(installation, "id", "unknown")),
        tokenizer_path=tokenizer_path,
        sample_rate=sample_rate,
        voices=voices,
        default_voice=default_voice if isinstance(default_voice, str) else None,
        metadata=metadata,
        installation=installation,
    )


def install_pretrained_model(
    ref: str,
    *,
    quality: str | None = None,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    refresh_catalog: bool = False,
    force_download: bool = False,
    catalog_url: str | None = None,
    progress: Any | None = None,
) -> ResolvedPhononModel:
    manager = _manager(cache_dir=cache_dir, offline=offline, catalog_url=catalog_url)
    try:
        installation = manager.install(
            normalize_ref(ref),
            quality=quality,
            refresh=refresh_catalog,
            force=force_download,
            progress=progress,
        )
    except Exception as exc:
        try:
            from onnxvoice.errors import OptionalDependencyError
        except ImportError:
            pass
        else:
            if isinstance(exc, OptionalDependencyError):
                raise ManagedModelDependencyError(
                    "Managed Phonon downloads require the Hugging Face transport; install "
                    "phononsynth[huggingface], phononsynth[cpu], or phononsynth[gpu]."
                ) from exc
        raise
    return installation_to_model(installation)


def open_installed_model(
    resolved: ResolvedPhononModel,
    *,
    providers: Sequence[Any] | str | None = None,
    provider_options: Sequence[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
    session_options: Any | None = None,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    catalog_url: str | None = None,
) -> Any:
    manager = _manager(cache_dir=cache_dir, offline=offline, catalog_url=catalog_url)
    try:
        return manager.open(
            resolved.installation,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
    except Exception as exc:
        if "phonon" in str(exc).lower() or "unsupported system" in str(exc).lower():
            raise OnnxVoiceContractError(
                "OnnxVoice must register a 'phonon' adapter before phononsynth can infer"
            ) from exc
        raise


def local_artifacts(bundle_dir: str | Path) -> tuple[dict[str, Path], dict[str, Any]]:
    root = Path(bundle_dir)
    config_path = root / "config.json"
    tokenizer_path = root / "tokenizer.json"
    if not config_path.is_file() or not tokenizer_path.is_file():
        raise UnsupportedModelError("local Phonon bundle requires config.json and tokenizer.json")
    import json

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnsupportedModelError("local Phonon config.json is invalid") from exc
    if not isinstance(config, Mapping):
        raise UnsupportedModelError("local Phonon config.json must contain an object")

    artifacts: dict[str, Path] = {"config": config_path, "tokenizer": tokenizer_path}
    for component in ("text_conditioner", "flow_lm_main", "flow_lm_flow", "mimi_decoder"):
        path = root / "onnx" / f"{component}.onnx"
        if not path.is_file():
            raise UnsupportedModelError(f"local Phonon bundle is missing {path.relative_to(root)}")
        artifacts[f"model:{component}"] = path

    raw_voices = config.get("voices")
    if not isinstance(raw_voices, Sequence) or isinstance(raw_voices, (str, bytes)):
        raise UnsupportedModelError("local Phonon config must declare voices")
    voices: list[str] = []
    for entry in raw_voices:
        name = entry.get("name") if isinstance(entry, Mapping) else None
        if not isinstance(name, str) or not name:
            raise UnsupportedModelError("local Phonon config contains an invalid voice")
        prompt = root / "voices" / f"{name}.bin"
        if not prompt.is_file():
            raise UnsupportedModelError(f"local Phonon bundle is missing voices/{name}.bin")
        artifacts[f"voice_prompt:{name}"] = prompt
        voices.append(name)
    return artifacts, {**dict(config), "_voices": voices}
