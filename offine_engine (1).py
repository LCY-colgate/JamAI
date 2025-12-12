import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.55")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # /jamAI
MAGENTA_RT_ROOT = os.path.join(THIS_DIR, "magenta-realtime")   # /jamAI/magenta-realtime

if MAGENTA_RT_ROOT not in sys.path:
    sys.path.insert(0, MAGENTA_RT_ROOT)

    
import numpy as np
import librosa

from constant import SAMPLE_RATE, CHUNK_SAMPLES
from engine import load_models
from InjectionStreamer import AudioInjectionStreamer
from config_store import make_default_config


def load_input_mono(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def chunkify_1d(y: np.ndarray, chunk_samples: int) -> list[np.ndarray]:
    if y.ndim != 1:
        raise ValueError(f"chunkify_1d expects mono (1D), got shape={y.shape}")

    chunks: list[np.ndarray] = []
    n = len(y)
    i = 0

    while i < n:
        c = y[i : i + chunk_samples]
        if len(c) < chunk_samples:
            c = np.pad(c, (0, chunk_samples - len(c)), mode="constant")
        chunks.append(c.astype(np.float32))
        i += chunk_samples

    return chunks


def save_wav(path: str, y_stereo: np.ndarray, sr: int = SAMPLE_RATE):
    y = np.asarray(y_stereo, dtype=np.float32)
    y = np.clip(y, -1.0, 1.0)

    import soundfile as sf
    sf.write(path, y, sr)


def run_offline_engine(
    output_wav: str,
    input_wav: str | None = None,
    prompt: str = "funk guitar groove",
    num_chunks: int = 32,
    beats_per_loop: int = 8,
    bpm: float = 120.0,
    warmup_chunks: int = 1,
):
    spectro, MRT = load_models()

    streamer = AudioInjectionStreamer(
        system=MRT,
        spectrostream_model=spectro,
        buffer_size=CHUNK_SAMPLES,
    )
    use_prerecorded_input = False  
    streamer.ui_config = make_default_config(
        initial_prompts=[prompt],
        beats_per_loop=beats_per_loop,
        use_prerecorded_input=use_prerecorded_input,
    )
    streamer.intro_loops = 0
    streamer.ui_config.set("metronome", False)
    streamer.ui_config.set("model_volume", 0.3)
    streamer.ui_config.set("model_feedback", 0.0)
    streamer.ui_config.set("temperature", 0.7)     
    streamer.ui_config.set("topk", 10)              
    streamer.ui_config.set("guidance_weight", 1.4) 
    streamer.ui_config.set("input_gap", 400)  

    streamer.use_prerecorded_input = use_prerecorded_input
    streamer.bpm = float(bpm)

    loop_seconds = beats_per_loop * 60.0 / streamer.bpm
    loop_samples = int(loop_seconds * SAMPLE_RATE)
    streamer.latency_samples = loop_samples

    streamer.metronome_audio = np.zeros(loop_samples + CHUNK_SAMPLES, dtype=np.float32)

    streamer.reset()
    streamer.injection_state.step = 0

    streamer.get_style_embedding(force_wait=True)

    if input_wav is None:
        in_chunks = [np.zeros(CHUNK_SAMPLES, dtype=np.float32) for _ in range(num_chunks)]
    else:
        mono = load_input_mono(input_wav, sr=SAMPLE_RATE)
        in_chunks = chunkify_1d(mono, CHUNK_SAMPLES)

        if len(in_chunks) < num_chunks:
            in_chunks += [np.zeros(CHUNK_SAMPLES, dtype=np.float32) for _ in range(num_chunks - len(in_chunks))]
        else:
            in_chunks = in_chunks[:num_chunks]

    # warmup：
    for _ in range(max(0, warmup_chunks)):
        _ = streamer(in_chunks[0])

    outs = []
    for t in range(num_chunks):
        target = 0.35
        ramp = 12  
        fb = target * min(1.0, t / ramp)
        streamer.ui_config.set("model_feedback", fb)
        x = in_chunks[t]
        y = streamer(x)  
        outs.append(y)

        if (t + 1) % 8 == 0:
            print(f"[offline_engine] generated {t+1}/{num_chunks} chunks")

    y_all = np.concatenate(outs, axis=0)
    save_wav(output_wav, y_all, sr=SAMPLE_RATE)
    print(f"[offline_engine] wrote: {output_wav}  shape={y_all.shape}")



if __name__ == "__main__":
    run_offline_engine(
        output_wav="offline_out5.wav",
        input_wav="03.wav",                 
        prompt="funk guitar groove",
        num_chunks=32,
        beats_per_loop=8,
        bpm=85.0,
        warmup_chunks=1,
    )

