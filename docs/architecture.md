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
