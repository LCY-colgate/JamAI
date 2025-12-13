# JamAI

JamAI (Colab-free) — Short-term README

This repo is my ongoing effort to remove Colab dependencies from **Magenta RealTime “Jam” mode** and run the pipeline locally:
- Offline streaming simulation (works)
- Real-time mic input / speaker output (next)
- Longer-term: “I play drums, AI plays other instruments” (separate-track architecture)

---

## What works right now

### 1) Colab-free offline generation loop
I implemented an offline engine that simulates streaming by:
- loading models (**SpectroStream** + **Magenta RT**)
- taking an input waveform (optional)
- chunkifying it into fixed-size blocks (currently `CHUNK_SAMPLES = 48000` = 1 second at 48kHz)
- feeding each chunk to `AudioInjectionStreamer`
- concatenating outputs and writing a WAV file

### 2) Basic config replacement (no Colab parameter UI)
Instead of Colab widgets, the pipeline uses a minimal config object (`make_default_config(...)`) assigned to `streamer.ui_config`.

### 3) Debug/stability findings (local)
During local runs I hit issues like:
- TensorFlow/JAX GPU memory contention (OOM / fragmentation warnings)
- `RuntimeError: Physical devices cannot be modified after being initialized`
- intermittent allocator warnings

Mitigations used when needed:
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- `TF_GPU_ALLOCATOR=cuda_malloc_async`
- avoid calling TF GPU memory config after TF is initialized

---

## Files 

offine_engine.py

Offline driver that chunk-feeds input and writes offline_out.wav.

InjectionStreamer.py, InjectionState.py

Core “audio injection” logic: maintains state, mixes inputs/feedback, encodes to tokens, calls model generation, handles crossfade, etc.

engine.py

Model wrapper: loads Magenta RT system + SpectroStream and exposes generate_chunk

config_store.py

Minimal replacement for Colab UI parameter store; provides make_default_config

crossfade.py, constant.py, prompt_types.py
Utility modules / constants.


