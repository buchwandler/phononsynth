from phononsynth import PhononVoice

with PhononVoice.from_pretrained("phonon-31m-en", quality="q4") as model:
    result = model.synthesize_prepared("Hello from PhononSynth.", seed=7)
    result.save_wav("phonon-example.wav")
