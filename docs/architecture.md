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

`phononsynth` owns only the text-facing convenience layer. OnnxVoice owns catalogs, installation, verification, provider selection, sessions, the runtime ABI, voice prompt loading, generation defaults, and diagnostics.
