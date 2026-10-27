# phononsynth

A user-facing Phonon text-to-speech frontend backed by OnnxVoice. PhononSynth loads the bundle's `tokenizer.json`, converts text to token IDs, and passes generation controls to OnnxVoice. OnnxVoice owns catalog discovery, direct-file downloads, integrity checks, cache management, and the streaming ONNX runtime.

## Install and synthesize

Install the CPU extra, inspect the live catalog, then save a WAV:

```bash
python -m pip install "phononsynth[cpu]"
phononsynth --list-models
phononsynth "Hello from Phonon." --model phonon-31m-en --quality q4 -o hello.wav
```

`phononsynth[cpu]` supplies the OnnxVoice CPU runtime as well as PhononSynth. Use `phononsynth[gpu]` for GPU runtime support. Normal Phonon bundle installation downloads direct per-file HTTPS release assets; it does **not** require Hugging Face. The optional `phononsynth[huggingface]` extra is only for workflows that actually use OnnxVoice's Hugging Face transport.

`--list-models` reads the active catalog and does not install model graphs. Its JSON includes each published model's active `release_version`, live quality choices, voices/default voice, languages, SPDX license, and commercial-use status. Available models and qualities can change with the catalog; use the values it reports rather than assuming every model supports every quality. The release version is intentionally reported from the catalog, not hardcoded here.

## Python API

```python
from phononsynth import PhononVoice

with PhononVoice.from_pretrained(
    "phonon-31m-en",
    quality="q4",
    cache_dir="~/.cache/phononsynth",
) as model:
    result = model.synthesize_prepared("Hello from Phonon.", seed=7)
    result.save_wav("hello.wav")
    print(result.metadata)  # model, quality, release/source provenance, token count, generation
```

Use the catalog's actual quality choices. Passing `quality=None` leaves quality resolution to OnnxVoice and preserves its catalog default. Set `offline=True` with the same cache directory to reopen an already installed model without network access.

## Streaming

Streaming forwards chunks directly from the runtime; it does not buffer the full utterance:

```python
from phononsynth import PhononVoice

with PhononVoice.from_pretrained("phonon-31m-en", quality="q4") as model:
    for chunk in model.stream_prepared("Streaming speech.", seed=7):
        consume(chunk.audio, chunk.sample_rate)
```

Each chunk carries float32 mono audio, its sample rate, frame index, EOS status, and EOS logit when provided by the runtime.

## Local bundle

`PhononVoice.from_local()` opens a task-01 canonical bundle directly, without registering it in the managed cache:

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

The config declares `runtime.kind` and `runtime.sample_rate`; voices may be declared as a sequence or a name-to-metadata mapping. The bundle must include a `.bin` prompt for every declared voice.

```python
with PhononVoice.from_local(bundle_dir="bundle") as model:
    model.synthesize_prepared("Local synthesis.").save_wav("local.wav")
```

## Generation controls and text limits

The CLI exposes `--voice`, `--duration-delta`, `--temperature`, `--seed`, `--eos-threshold`, `--frames-after-eos`, `--min-frames-before-eos`, and `--max-frames`. The Python API accepts these individually or through `SynthesisConfig`, but not both in one call.

PhononSynth uses the installed bundle tokenizer and validates empty text before inference. The current Phonon runtime contract does not declare a maximum input-token limit, so `measure_prepared()` reports a real token count with `max_tokens=None` and `fits=None`. PhononSynth does not invent a limit or automatically split long text.

## Model licenses

The Python package license and model-artifact licenses are separate. Check each model's live SPDX and `commercial_use` values in `phononsynth --list-models` before downloading or using its artifacts. The 40m checkpoints are **CC-BY-NC-4.0 and non-commercial only**. This warning is not a grant of commercial rights; comply with the model license and required notices.

## Scope

PhononSynth is a tokenizer/frontend for Phonon's streaming generation contract. It does not implement voice cloning, pronunciation overrides, word timings, text splitting, or long-form orchestration.
