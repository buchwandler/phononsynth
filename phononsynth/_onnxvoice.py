from __future__ import annotations

import inspect
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


def _require_phonon_contract(module: Any) -> None:
    """Fail early when OnnxVoice lacks the released Phonon manager/adapter API."""
    manager_type = getattr(module, "OnnxVoice", None)
    missing: list[str] = []
    if not callable(manager_type):
        missing.append("OnnxVoice manager")
    else:
        required = {
            "OnnxVoice constructor": (manager_type, ("cache_dir", "catalog_sources", "offline")),
            "OnnxVoice.install": (
                getattr(manager_type, "install", None),
                ("quality", "refresh", "force", "progress"),
            ),
            "OnnxVoice.open": (
                getattr(manager_type, "open", None),
                ("providers", "provider_options", "session_options"),
            ),
            "OnnxVoice.open_local": (
                getattr(manager_type, "open_local", None),
                ("system", "artifacts", "providers", "provider_options", "session_options"),
            ),
        }
        for label, (callable_obj, parameters) in required.items():
            if not callable(callable_obj):
                missing.append(label)
                continue
            try:
                signature = inspect.signature(callable_obj)
            except (TypeError, ValueError):
                missing.append(f"inspectable {label} signature")
                continue
            accepts_keywords = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            absent = [name for name in parameters if name not in signature.parameters]
            if absent and not accepts_keywords:
                missing.append(f"{label} parameters: {', '.join(absent)}")
    try:
        from onnxvoice.systems import get_adapter

        get_adapter("phonon")
    except Exception as exc:
        missing.append(f"registered Phonon adapter ({exc})")
    if missing:
        details = "; ".join(missing)
        raise OnnxVoiceContractError(
            "Installed OnnxVoice does not provide the required Phonon contract "
            f"({details}). Install a compatible release: onnxvoice>=0.2.15,<0.3."
        )


def _optional_dependency_error(exc: Exception) -> ManagedModelDependencyError | None:
    try:
        from onnxvoice.errors import OptionalDependencyError
    except ImportError:
        return None
    if not isinstance(exc, OptionalDependencyError):
        return None

    detail = str(exc)
    normalized = detail.casefold()
    if "huggingface" in normalized or "hugging face" in normalized:
        hint = "Install 'phononsynth[huggingface]' only when using Hugging Face downloads."
    elif any(
        token in normalized for token in ("onnxruntime", "onnx runtime", "execution provider")
    ):
        hint = "For CPU inference, install 'phononsynth[cpu]'."
    else:
        hint = ""
    suffix = f" {hint}" if hint else ""
    return ManagedModelDependencyError(
        f"OnnxVoice reports a missing optional dependency: {detail}.{suffix}"
    )


def normalize_ref(value: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError("model reference cannot be empty")
    return value if ":" in value else f"phonon:{value}"


def _manager(*, cache_dir: str | Path | None, offline: bool, catalog_url: str | None) -> Any:
    module = _module()
    _require_phonon_contract(module)
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
        dependency_error = _optional_dependency_error(exc)
        if dependency_error is not None:
            raise dependency_error from exc
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
    return manager.open(
        resolved.installation,
        providers=providers,
        provider_options=provider_options,
        session_options=session_options,
    )


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

    runtime_config = config.get("runtime", {})
    if runtime_config is not None and not isinstance(runtime_config, Mapping):
        raise UnsupportedModelError("local Phonon config runtime must be an object")
    if isinstance(runtime_config, Mapping) and "sample_rate" in runtime_config:
        sample_rate = runtime_config["sample_rate"]
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise UnsupportedModelError(
                "local Phonon runtime.sample_rate must be a positive integer"
            )

    raw_voices = config.get("voices")
    if isinstance(raw_voices, Mapping):
        declarations = list(raw_voices.items())
        if any(not isinstance(entry, Mapping) for _, entry in declarations):
            raise UnsupportedModelError("local Phonon mapping voices must contain objects")
    elif isinstance(raw_voices, Sequence) and not isinstance(raw_voices, (str, bytes)):
        declarations = [
            (entry.get("name"), entry) for entry in raw_voices if isinstance(entry, Mapping)
        ]
        if len(declarations) != len(raw_voices):
            raise UnsupportedModelError("local Phonon config contains an invalid voice")
    else:
        raise UnsupportedModelError("local Phonon config must declare voices")

    voices: list[str] = []
    for name, _entry in declarations:
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or "\x00" in name
            or Path(name).is_absolute()
        ):
            raise UnsupportedModelError("local Phonon config contains an unsafe voice name")
        if name in voices:
            raise UnsupportedModelError(f"local Phonon config declares duplicate voice {name!r}")
        prompt = root / "voices" / f"{name}.bin"
        if not prompt.is_file():
            raise UnsupportedModelError(f"local Phonon bundle is missing voices/{name}.bin")
        artifacts[f"voice_prompt:{name}"] = prompt
        voices.append(name)

    default_voice = config.get("default_voice")
    if default_voice is not None and default_voice not in voices:
        raise UnsupportedModelError("local Phonon default_voice must be declared in voices")
    return artifacts, {**dict(config), "_voices": tuple(voices)}
