from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ._onnxvoice import _manager
from .errors import CatalogDiscoveryError, CatalogUnavailableError, PhononSynthError


@dataclass(frozen=True, slots=True)
class DescribedVoice:
    id: str
    display_name: str
    gender: str = "unknown"
    language: str = ""
    locale: str = ""
    languages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    id: str
    display_name: str
    languages: tuple[str, ...]
    quality_choices: tuple[str, ...]
    sample_rate: int
    voices: tuple[DescribedVoice, ...]
    default_voice: str | None
    source_revision: str | None
    license_spdx: str | None
    commercial_use: bool | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def voice_ids(self) -> tuple[str, ...]:
        return tuple(voice.id for voice in self.voices)


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


def _voice(record: Any) -> DescribedVoice:
    metadata = getattr(record, "metadata", None)
    return DescribedVoice(
        id=str(getattr(record, "voice_id", "")),
        display_name=str(
            getattr(metadata, "language_label", "") or getattr(record, "voice_id", "")
        ),
        gender=str(getattr(metadata, "gender", "unknown") or "unknown"),
        language=str(getattr(metadata, "language", "") or ""),
        locale=str(getattr(metadata, "locale", "") or ""),
        languages=_strings(getattr(record, "languages", ())),
    )


def _model(item: Any, records: tuple[Any, ...]) -> DiscoveredModel:
    metadata = dict(getattr(item, "metadata", {}) or {})
    item_id = str(getattr(item, "id", ""))
    matching = tuple(
        record
        for record in records
        if getattr(getattr(record, "catalog_item", None), "id", None) == item_id
    )
    raw_details = metadata.get("voice_details")
    detail_by_id: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw_details, Sequence) and not isinstance(raw_details, (str, bytes)):
        detail_by_id = {
            str(detail.get("id")): detail
            for detail in raw_details
            if isinstance(detail, Mapping) and isinstance(detail.get("id"), str)
        }
    voices_list: list[DescribedVoice] = []
    for record in matching:
        described = _voice(record)
        detail = detail_by_id.get(described.id, {})
        voices_list.append(
            DescribedVoice(
                id=described.id,
                display_name=str(detail.get("display_name") or detail.get("name") or described.id),
                gender=described.gender,
                language=described.language,
                locale=described.locale,
                languages=described.languages,
            )
        )
    voices = tuple(voices_list)
    if not voices:
        rows: list[DescribedVoice] = []
        for detail in detail_by_id.values():
            voice_id = str(detail["id"])
            rows.append(
                DescribedVoice(
                    id=voice_id,
                    display_name=str(detail.get("display_name") or detail.get("name") or voice_id),
                    gender=str(detail.get("gender") or "unknown"),
                    language=str(detail.get("language") or ""),
                    locale=str(detail.get("locale") or ""),
                    languages=_strings(detail.get("languages") or (detail.get("language"),)),
                )
            )
        voices = tuple(rows)
    license_info = metadata.get("license")
    license_info = license_info if isinstance(license_info, Mapping) else {}
    commercial = license_info.get("commercial_use", metadata.get("commercial_use"))
    if not isinstance(commercial, bool):
        commercial = None
    return DiscoveredModel(
        id=item_id,
        display_name=str(metadata.get("name") or item_id),
        languages=_strings(metadata.get("languages") or metadata.get("language_codes")),
        quality_choices=_strings(metadata.get("quality_profiles")),
        sample_rate=int(getattr(item, "sample_rate", None) or 24000),
        voices=voices,
        default_voice=getattr(item, "default_voice", None),
        source_revision=metadata.get("source_revision")
        if isinstance(metadata.get("source_revision"), str)
        else None,
        license_spdx=license_info.get("spdx")
        if isinstance(license_info.get("spdx"), str)
        else metadata.get("license_spdx")
        if isinstance(metadata.get("license_spdx"), str)
        else None,
        commercial_use=commercial,
        metadata=metadata,
    )


def discover_models(
    *,
    language: str | None = None,
    offline: bool = False,
    refresh: bool = False,
    cache_dir: str | Path | None = None,
    catalog_url: str | None = None,
) -> tuple[DiscoveredModel, ...]:
    """Discover Phonon catalog targets without installing or opening them."""
    try:
        manager = _manager(cache_dir=cache_dir, offline=offline, catalog_url=catalog_url)
        items = manager.list("phonon", language=language, refresh=refresh)
        records = tuple(manager.list_voices("phonon", language=language, refresh=False))
    except CatalogUnavailableError:
        raise
    except CatalogDiscoveryError as exc:
        raise CatalogUnavailableError(str(exc)) from exc
    except PhononSynthError:
        raise
    except Exception as exc:
        raise CatalogUnavailableError(f"Phonon model catalog is unavailable: {exc}") from exc
    return tuple(_model(item, records) for item in items if item.system == "phonon")
