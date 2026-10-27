from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._onnxvoice import (
    _module,
    _optional_dependency_error,
    _require_phonon_contract,
    install_pretrained_model,
    local_artifacts,
    open_installed_model,
)
from ._tokenizer import encode_text, load_tokenizer
from .config import SynthesisConfig
from .errors import EmptyTextError, ModelInferenceError, PhononSynthError, UnsupportedModelError
from .types import StreamChunk, SynthesisMeasure, SynthesisResult, VoiceInfo


class PhononVoice:
    def __init__(
        self,
        *,
        runtime: Any,
        tokenizer: Any,
        metadata: Mapping[str, Any],
        model_ref: str | None,
        model_id: str | None = None,
        voices: Sequence[str] = (),
        default_voice: str | None = None,
        sample_rate: int = 24000,
    ) -> None:
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.metadata = dict(metadata)
        self.model_ref = model_ref
        self.model_id = model_id
        self.sample_rate = int(sample_rate)
        self._voices = tuple(str(value) for value in voices)
        self.default_voice = default_voice or (self._voices[0] if self._voices else None)
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        model: str = "phonon-31m-en",
        *,
        quality: str | None = None,
        cache_dir: str | Path | None = None,
        offline: bool = False,
        refresh_catalog: bool = False,
        force_download: bool = False,
        catalog_url: str | None = None,
        providers: Sequence[Any] | str | None = None,
        provider_options: Sequence[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
        session_options: Any | None = None,
        progress: Any | None = None,
    ) -> PhononVoice:
        resolved = install_pretrained_model(
            model,
            quality=quality,
            cache_dir=cache_dir,
            offline=offline,
            refresh_catalog=refresh_catalog,
            force_download=force_download,
            catalog_url=catalog_url,
            progress=progress,
        )
        runtime = open_installed_model(
            resolved,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
            cache_dir=cache_dir,
            offline=offline,
            catalog_url=catalog_url,
        )
        return cls(
            runtime=runtime,
            tokenizer=load_tokenizer(resolved.tokenizer_path),
            metadata=resolved.metadata,
            model_ref=resolved.ref,
            model_id=resolved.model_id,
            voices=resolved.voices,
            default_voice=resolved.default_voice,
            sample_rate=resolved.sample_rate,
        )

    @classmethod
    def from_local(
        cls,
        *,
        bundle_dir: str | Path,
        providers: Sequence[Any] | str | None = None,
        provider_options: Sequence[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
        session_options: Any | None = None,
    ) -> PhononVoice:
        artifacts, config = local_artifacts(bundle_dir)
        onnxvoice = _module()
        _require_phonon_contract(onnxvoice)
        runtime = onnxvoice.OnnxVoice.open_local(
            system="phonon",
            artifacts=artifacts,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
        runtime_config = config.get("runtime") if isinstance(config.get("runtime"), Mapping) else {}
        voices = tuple(config.get("_voices", ()))
        default_voice = config.get("default_voice")
        raw_sample_rate = runtime_config.get("sample_rate")
        if raw_sample_rate is None:
            sample_rate = 24000
        elif (
            isinstance(raw_sample_rate, bool)
            or not isinstance(raw_sample_rate, int)
            or raw_sample_rate <= 0
        ):
            raise UnsupportedModelError(
                "local Phonon runtime.sample_rate must be a positive integer"
            )
        else:
            sample_rate = raw_sample_rate
        return cls(
            runtime=runtime,
            tokenizer=load_tokenizer(artifacts["tokenizer"]),
            metadata={"runtime": dict(runtime_config), "local_bundle": str(Path(bundle_dir))},
            model_ref=None,
            model_id=None,
            voices=voices,
            default_voice=default_voice if isinstance(default_voice, str) else None,
            sample_rate=sample_rate,
        )

    @property
    def available_voices(self) -> tuple[str, ...]:
        return self._voices

    @property
    def voices(self) -> tuple[VoiceInfo, ...]:
        details = self.metadata.get("voice_details")
        by_id: dict[str, Mapping[str, Any]] = {}
        if isinstance(details, Sequence) and not isinstance(details, (str, bytes)):
            by_id = {
                str(row.get("id")): row
                for row in details
                if isinstance(row, Mapping) and isinstance(row.get("id"), str)
            }
        rows: list[VoiceInfo] = []
        for name in self.available_voices:
            detail = by_id.get(name, {})
            rows.append(
                VoiceInfo(
                    name=name,
                    display_name=str(detail.get("display_name") or detail.get("name") or name),
                    language=str(detail.get("language") or ""),
                    locale=str(detail.get("locale") or ""),
                    gender=str(detail.get("gender") or "unknown"),
                )
            )
        return tuple(rows)

    def _token_ids(self, text: str) -> tuple[int, ...]:
        if self._closed:
            raise RuntimeError("PhononVoice is closed")
        if not isinstance(text, str) or not text.strip():
            raise EmptyTextError("text must contain speakable content")
        return encode_text(self.tokenizer, text)

    def measure_prepared(self, text: str) -> SynthesisMeasure:
        token_ids = self._token_ids(text)
        return SynthesisMeasure(
            token_count=len(token_ids),
            max_tokens=None,
            fits=None,
            model_id=self.model_id,
        )

    @staticmethod
    def _config(config: SynthesisConfig | None, overrides: dict[str, Any]) -> SynthesisConfig:
        if config is not None and any(value is not None for value in overrides.values()):
            raise ValueError("Pass either config or individual generation overrides, not both")
        if config is not None:
            return config.validated()
        return SynthesisConfig(**overrides).validated()

    def _voice(self, voice: str | None) -> str:
        selected = voice or self.default_voice
        if selected is None:
            raise ValueError("No default Phonon voice is available; pass voice= explicitly")
        if self._voices and selected not in self._voices:
            available = ", ".join(self._voices)
            raise ValueError(f"Unknown Phonon voice {selected!r}; available: {available}")
        return selected

    def synthesize_prepared(
        self,
        text: str,
        *,
        voice: str | None = None,
        config: SynthesisConfig | None = None,
        duration_delta: float | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        eos_threshold: float | None = None,
        frames_after_eos: int | None = None,
        min_frames_before_eos: int | None = None,
        max_frames: int | None = None,
    ) -> SynthesisResult:
        token_ids = self._token_ids(text)
        selected_voice = self._voice(voice)
        generation = self._config(
            config,
            {
                "duration_delta": duration_delta,
                "temperature": temperature,
                "seed": seed,
                "eos_threshold": eos_threshold,
                "frames_after_eos": frames_after_eos,
                "min_frames_before_eos": min_frames_before_eos,
                "max_frames": max_frames,
            },
        )
        try:
            result = self.runtime.infer(
                token_ids,
                voice=selected_voice,
                **generation.runtime_kwargs(),
            )
        except PhononSynthError:
            raise
        except Exception as exc:
            dependency_error = _optional_dependency_error(exc)
            if dependency_error is not None:
                raise ModelInferenceError(
                    f"Phonon model inference failed: {dependency_error}"
                ) from exc
            raise ModelInferenceError("Phonon model inference failed") from exc
        audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio must be finite")
        sample_rate = int(getattr(result, "sample_rate", 0) or self.sample_rate)
        sample_rate = int(getattr(result, "sample_rate", 0) or self.sample_rate)
        runtime_metadata = getattr(result, "metadata", None)
        metadata = dict(runtime_metadata) if isinstance(runtime_metadata, Mapping) else {}
        selected_quality = self.metadata.get("selected_quality")
        selected_quality = selected_quality if isinstance(selected_quality, str) else None
        release_tags = self.metadata.get("release_tags")
        source_revisions = self.metadata.get("source_revisions")
        release_versions = self.metadata.get("release_versions")
        release_tag = (
            release_tags.get(selected_quality)
            if isinstance(release_tags, Mapping) and selected_quality is not None
            else None
        ) or self.metadata.get("release_tag")
        source_revision = (
            source_revisions.get(selected_quality)
            if isinstance(source_revisions, Mapping) and selected_quality is not None
            else None
        ) or self.metadata.get("source_revision")
        release_version = self.metadata.get("release_version")
        if not isinstance(release_version, str) and isinstance(release_versions, Mapping):
            release_version = release_versions.get(selected_quality)
        frontend_metadata = {
            "model_id": self.model_id,
            "selected_quality": selected_quality,
            "release_version": release_version,
            "release_tag": release_tag,
            "source_revision": source_revision,
            "token_count": len(token_ids),
            "generation": generation.runtime_kwargs(),
        }
        for key, value in frontend_metadata.items():
            if value is not None:
                metadata.setdefault(key, value)
        return SynthesisResult(
            audio=audio,
            sample_rate=sample_rate,
            voice=selected_voice,
            model_ref=self.model_ref,
            metadata=metadata,
        )

    synthesize_text = synthesize_prepared

    def stream_prepared(
        self,
        text: str,
        *,
        voice: str | None = None,
        config: SynthesisConfig | None = None,
        duration_delta: float | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        eos_threshold: float | None = None,
        frames_after_eos: int | None = None,
        min_frames_before_eos: int | None = None,
        max_frames: int | None = None,
    ) -> Iterator[StreamChunk]:
        token_ids = self._token_ids(text)
        selected_voice = self._voice(voice)
        generation = self._config(
            config,
            {
                "duration_delta": duration_delta,
                "temperature": temperature,
                "seed": seed,
                "eos_threshold": eos_threshold,
                "frames_after_eos": frames_after_eos,
                "min_frames_before_eos": min_frames_before_eos,
                "max_frames": max_frames,
            },
        )
        try:
            for chunk in self.runtime.stream(
                token_ids,
                voice=selected_voice,
                **generation.runtime_kwargs(),
            ):
                yield StreamChunk(
                    audio=chunk.audio,
                    sample_rate=int(chunk.sample_rate),
                    frame_index=int(chunk.frame_index),
                    eos_detected=bool(chunk.eos_detected),
                    eos_logit=getattr(chunk, "eos_logit", None),
                )
        except PhononSynthError:
            raise
        except Exception as exc:
            dependency_error = _optional_dependency_error(exc)
            if dependency_error is not None:
                raise ModelInferenceError(
                    f"Phonon streaming inference failed: {dependency_error}"
                ) from exc
            raise ModelInferenceError("Phonon streaming inference failed") from exc

    stream_text = stream_prepared

    def diagnostics(self) -> Any:
        diagnostics = getattr(self.runtime, "diagnostics", None)
        return diagnostics() if callable(diagnostics) else diagnostics

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self.runtime, "close", None)
        try:
            if callable(close):
                close()
        finally:
            self._closed = True

    def __enter__(self) -> PhononVoice:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
