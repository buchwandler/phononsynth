from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .discovery import discover_models
from .voice import PhononVoice


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize one prepared Phonon request")
    parser.add_argument("text", nargs="?")
    parser.add_argument("--model", default="phonon-31m-en")
    parser.add_argument("--quality")
    parser.add_argument("--voice")
    parser.add_argument("--duration-delta", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eos-threshold", type=float)
    parser.add_argument("--frames-after-eos", type=int)
    parser.add_argument("--min-frames-before-eos", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("-o", "--output", default="phonon.wav")
    parser.add_argument("--cache-dir")
    parser.add_argument("--catalog-url")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_models:
        models = discover_models(
            offline=args.offline,
            cache_dir=args.cache_dir,
            catalog_url=args.catalog_url,
        )
        print(
            json.dumps(
                [
                    {
                        "id": model.id,
                        "release_version": model.release_version,
                        "languages": model.languages,
                        "qualities": model.quality_choices,
                        "voices": model.voice_ids,
                        "default_voice": model.default_voice,
                        "license": model.license_spdx,
                        "commercial_use": model.commercial_use,
                    }
                    for model in models
                ],
                indent=2,
            )
        )
        return 0
    if not args.text:
        raise SystemExit("text is required unless --list-models is used")
    with PhononVoice.from_pretrained(
        args.model,
        quality=args.quality,
        cache_dir=args.cache_dir,
        catalog_url=args.catalog_url,
        offline=args.offline,
    ) as model:
        result = model.synthesize_prepared(
            args.text,
            voice=args.voice,
            duration_delta=args.duration_delta,
            temperature=args.temperature,
            seed=args.seed,
            eos_threshold=args.eos_threshold,
            frames_after_eos=args.frames_after_eos,
            min_frames_before_eos=args.min_frames_before_eos,
            max_frames=args.max_frames,
        )
        result.save_wav(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
