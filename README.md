# phononsynth (MVP)

A small text-to-speech frontend for **Phonon**, designed to feel like `kittensynth` while keeping Phonon's streaming/generation controls visible.

It delegates model catalogs, installation, cache integrity, ONNX Runtime sessions, the four-graph Phonon ABI, and bundled voice prompts to `onnxvoice`. `phononsynth` adds the missing text layer by loading the bundle's standard `tokenizer.json`, plus a compact Python/CLI interface.

## Install

```bash
python -m pip install -e '.[cpu]'
```

The base package does not force a CPU or GPU ONNX Runtime provider. The published Phonon bundle catalog uses direct per-file release URLs, so managed runtime installation does not require the Hugging Face transport. The `huggingface` extra is retained only as an optional OnnxVoice capability.

## Managed model

```python
from phononsynth import PhononVoice

with PhononVoice.from_pretrained("phonon-31m-en", quality="q4") as model:
    result = model.synthesize_prepared(
        "Hello from PhononSynth.",
        voice=model.default_voice,
        temperature=0.8,
        seed=7,
    )
    result.save_wav("hello.wav")
```

`quality=None` lets OnnxVoice select its catalog default (currently fp32 when present). A cold managed run may use the network.

## Streaming

```python
from phononsynth import PhononVoice

with PhononVoice.from_pretrained("phonon-31m-en", quality="q4") as model:
    for chunk in model.stream_prepared("Streaming speech.", seed=7):
        consume(chunk.audio, chunk.sample_rate)
```

## Local exported bundle

The local layout is the bundle layout produced by `phonon-onnx-bundles`:

```text
bundle/
  config.json
  tokenizer.json
  onnx/
    text_conditioner.onnx
    flow_lm_main.onnx
    flow_lm_flow.onnx
    mimi_decoder.onnx
  voices/
    <voice>.bin
```

Open it directly, without registering it in the OnnxVoice cache:

```python
from phononsynth import PhononVoice

with PhononVoice.from_local(bundle_dir="bundle") as model:
    model.synthesize_prepared("Local synthesis.").save_wav("local.wav")
```

## Discovery

```python
from phononsynth import discover_models

for model in discover_models(language="en"):
    print(model.id, model.quality_choices, model.voice_ids, model.license_spdx)
```

Discovery only reads catalog metadata; it does not install or open model graphs.

## CLI

```bash
phononsynth "Hello." --model phonon-31m-en --quality q4 -o hello.wav
phononsynth --list-models
```

Useful controls are available directly: `--voice`, `--duration-delta`, `--temperature`, `--seed`, `--eos-threshold`, `--frames-after-eos`, `--min-frames-before-eos`, and `--max-frames`.

## License boundary

The Python package and model artifacts have separate licensing concerns. In the supplied bundle registry, `phonon-31m-en` is Apache-2.0, while the 40m checkpoints are CC BY-NC 4.0 and non-commercial. `discover_models()` preserves the model license/commercial-use metadata so applications can surface it before installation or use.

This MVP intentionally does not implement voice cloning, pronunciation overrides, word timings, text splitting, or automatic long-form chunking. The Phonon runtime contract does not declare a maximum input-token limit, so `measure_prepared()` reports an exact token count with `max_tokens=None` and `fits=None`.
