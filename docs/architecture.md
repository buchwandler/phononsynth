# Architecture

```text
prepared text
    |
    v
bundle tokenizer.json (Hugging Face tokenizers)
    |
    v
model token IDs
    |
    v
onnxvoice PhononAdapter
    |-- text_conditioner.onnx
    |-- flow_lm_main.onnx
    |-- flow_lm_flow.onnx
    `-- mimi_decoder.onnx
    |
    v
24 kHz float32 PCM / streaming frames
```

`phononsynth` owns the text-facing convenience layer and fills generation controls omitted by older bundles with Phonon-compatible defaults. OnnxVoice owns catalogs, installation, verification, provider selection, sessions, the runtime ABI, voice prompt loading, bundle generation defaults, and diagnostics.

PhononSynth measures prepared text with the bundle tokenizer only and does not run acoustic inference during measurement. The current Phonon runtime contract declares no finite text-token maximum, so request API v2 returns the exact token count with `max_tokens=None` and `fits=True` after successful measurement. The fit result means the request satisfies all declared PhononSynth input-capacity constraints, not that every inference will avoid resource failures. Long-form splitting and orchestration remain upstream; PhononSynth does not automatically split text.
