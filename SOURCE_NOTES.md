# Source-derived MVP decisions

This MVP was derived from the supplied repository snapshots rather than from a separate Phonon implementation.

- **Kittensynth-shaped interface:** mirrors the useful surface of `KittenVoice`: managed/local constructors, `available_voices`, `voices`, `measure_prepared`, `synthesize_prepared`, a compatibility `synthesize_text` alias, diagnostics, context-manager lifecycle, result WAV saving, discovery, and CLI packaging.
- **Phonon-specific runtime:** does not emulate Kitten's `speed`/style controls. It forwards Phonon's declared controls: duration delta, temperature, seed, EOS threshold/tail, minimum frames, and maximum frames.
- **Streaming:** Phonon exposes native streaming PCM, so the wrapper adds `stream_prepared` / `stream_text` rather than collecting all output unconditionally.
- **Tokenizer boundary:** OnnxVoice's Phonon adapter accepts already-tokenized integer IDs. Phonon bundles require `tokenizer.json`, so this package uses Hugging Face `tokenizers` to convert prepared text to model IDs before calling the adapter.
- **Local layout:** fixed to the supplied Phonon bundle ABI: four ONNX graphs, `config.json`, `tokenizer.json`, and `voices/<name>.bin` prompts.
- **Input capacity:** the OnnxVoice Phonon options contract reports no declared maximum input-token limit. `measure_prepared` therefore reports the exact tokenizer count but leaves `max_tokens` and `fits` unknown.
- **Licensing:** discovery preserves catalog model-license metadata. The supplied registry marks 31m English Apache-2.0/commercial and 40m variants CC-BY-NC-4.0/non-commercial.
