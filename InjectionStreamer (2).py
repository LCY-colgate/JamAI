from constant import SAMPLE_RATE, CHUNK_SECONDS, CHUNK_SAMPLES, MIX_PREFIX_FRAMES, LEFT_EDGE_FRAMES_TO_REMOVE
from crossfade import AudioFade
from InjectionState import AudioInjectionState
from config_store import StaticConfig
import prompt_types
import numpy as np
import concurrent.futures
import functools
import resampy
from magenta_rt import audio as audio_lib
from magenta_rt import system
from magenta_rt import spectrostream

class AudioInjectionStreamer:
  """Audio streamer class for Magenta RT model with Audio Injection.

  This class holds a pretrained Magenta RT model, a cross fade state, a
  generation state, audio injection state, and an asynchronous executor to
  handle prompt embedding without interrupting the audio thread.

  Args:
    system: A MagentaRTBase instance.
  """

  def __init__(
      self,
      spectrostream_model,
      system: system.MagentaRTBase,
      sample_rate: int = SAMPLE_RATE,
      num_channels: int = 2,
      buffer_size: int = 2 * SAMPLE_RATE,
      extra_buffering: int = 0
  ):      
    config = system.config
    self.system = system
    self.spectrostream_model = spectrostream_model
    
    self.audio_streamer = None
    self.sample_rate = sample_rate
    self.num_channels = num_channels
    self.buffer_size = buffer_size
    self.extra_buffering = extra_buffering
    #added
    self.ui_config: StaticConfig | None = None
    self.use_prerecorded_input: bool = False
    self.input_audio = None          
    self.metronome_audio = None      
    self.bpm: float = 120.0
    self.intro_loops: int = 4
    self.latency_samples: int | None = None  # live
    self.fade = AudioFade(
        chunk_size=int(config.codec_sample_rate * config.crossfade_length),
        num_chunks=1,
        stereo=True
    )
    self.state = None
    self.executor = concurrent.futures.ThreadPoolExecutor()
    context_seconds = config.context_length
    context_frames = int(context_seconds * config.codec_frame_rate)
    context_samples = int(context_seconds * SAMPLE_RATE)
    self.injection_state = AudioInjectionState(
        context_tokens_orig=np.zeros(
            (context_frames, config.decoder_codec_rvq_depth),
            dtype=np.int32
        ),
        all_inputs=np.zeros(
            (context_samples, 2)
            if self.use_prerecorded_input else (context_samples,),
            dtype=np.float32
        ),
        all_outputs=np.zeros((context_samples, 2), dtype=np.float32),
        step=-1,  # This will be 0 after the warmup call.
    )

  @property
  def warmup(self):
    """Returns whether to warm up the audio streamer."""
    return True

  def on_stream_start(self):
    """Called when the UI starts streaming."""
    self.get_style_embedding(force_wait=False)
    self.get_style_embedding(force_wait=True)
    if self.audio_streamer is not None:
      self.audio_streamer.reset_ring_buffer()

  def on_stream_stop(self):
    """Called when the UI stops streaming."""
    pass

  def reset(self):
    self.state = None
    self.fade.reset()
    self.embed_style.cache_clear()
    if self.audio_streamer is not None:
      self.audio_streamer.reset_ring_buffer()

  def start(self):
    #to do
    raise NotImplementedError("Realtime streaming not implemented yet (no colab_utils)")
    """
    self.audio_streamer = colab_utils.AudioStreamer(
        self,
        rate=self.sample_rate,
        buffer_size=self.buffer_size,
        enable_input=True,
        warmup=self.warmup,
        raw_input_audio=True,
        enable_automatic_gain_control_on_input=True,
        num_output_channels=self.num_channels,
        additional_buffered_samples=self.extra_buffering,
        start_streaming_callback=self.on_stream_start,
        stop_streaming_callback=self.on_stream_stop,
    )
    self.reset()
    """

  def stop(self):
    self.executor.shutdown(wait=True)
    if self.audio_streamer is not None:
        del self.audio_streamer
    self.audio_streamer = None
      
  def global_ui_params(self):
    #return colab_utils.Parameters.get_values()
    if self.ui_config is None:
        raise RuntimeError("ui_config is not set on AudioInjectionStreamer.")
    return self.ui_config.get_values()

  @functools.cache
  def embed_style(self, style: str):
    return self.executor.submit(self.system.embed_style, style)

  @functools.cache
  def embed_16k_audio(self, audio: tuple[float]):
    """Embed 16k audio asyncronously, returning a future."""
    audio = audio_lib.Waveform(np.asarray(audio), 16000)
    return self.executor.submit(self.system.embed_style, audio)

  def embed_48k_audio(self, audio: tuple[float]):
    """Embed 48k audio asyncronously, returning an embedding."""
    resampled_audio = resampy.resample(np.asarray(audio), 48000, 16000)
    return self.embed_16k_audio(tuple(resampled_audio)).result()

  def get_style_embedding(self, force_wait: bool = False):
    prompts = self.get_prompts()
    weighted_embedding = np.zeros((768,), dtype=np.float32)
    total_weight = 0.0
    for prompt_value, prompt_weight in prompts:
      match type(prompt_value):
        case prompt_types.TextPrompt:
          if not prompt_value:
            continue
          embedding = self.embed_style(prompt_value)

        case prompt_types.AudioPrompt:
          embedding = self.embed_16k_audio(tuple(prompt_value.value))

        case prompt_types.EmbeddingPrompt:
          embedding = prompt_value.value

        case _:
          raise ValueError(f"Unsupported prompt type: {type(prompt_value)}")

      if isinstance(embedding, concurrent.futures.Future):
        if force_wait:
          embedding.result()

        if not embedding.done():
          continue

        embedding = embedding.result()

      weighted_embedding += embedding * prompt_weight
      total_weight += prompt_weight

    if total_weight > 0:
      weighted_embedding /= total_weight

    return weighted_embedding

  def get_prompts(self):
    params = self.global_ui_params()
    num_prompts = sum(map(lambda s: "prompt_value" in s, params.keys()))
    prompts = []
    for i in range(num_prompts):
      prompt_weight = params[f"prompt_weight_{i}"]
      prompt_value = params[f"prompt_value_{i}"]

      if prompt_value is None or not prompt_weight:
        continue

      match type(prompt_value):
        case prompt_types.TextPrompt:
          text = prompt_value.strip()
          prompt_value = prompt_types.TextPrompt(text)
        case prompt_types.AudioPrompt:
          pass
        case prompt_types.EmbeddingPrompt:
          pass
        case _:
          raise ValueError(f"Unsupported prompt type: {type(prompt_value)}")

      prompts.append((prompt_value, prompt_weight))
    return prompts

  def generate(self, ui_params, inputs):
    if self.use_prerecorded_input:
      assert self.input_audio is not None, (
            "To use prerecorded input, please set streamer.input_audio first."
        )
      start = (self.injection_state.step * CHUNK_SAMPLES) % (len(self.input_audio) - CHUNK_SAMPLES)
      end = start + CHUNK_SAMPLES
      inputs = self.input_audio[start:end]
      inputs_mono = np.mean(inputs, axis=-1)
    else:
      inputs_mono = inputs
    """
    if LIVE_AUDIO_PROMPT is not None:
      # Update live audio prompt with latest inputs.
      LIVE_AUDIO_PROMPT.update_audio_input(
          inputs_mono if use_prerecorded_input else inputs)
    """
      
    # Add this input chunk to the end of `all_inputs`.
    self.injection_state.all_inputs = np.concatenate(
        [self.injection_state.all_inputs, inputs], axis=0
    )

    # Pass an extra prefix of mixed audio to the encoder so we can throw away
    # the earliest frames, which have edge artifacts.
    mix_samples = (CHUNK_SAMPLES + MIX_PREFIX_FRAMES * self.system.config.frame_length_samples)

    # Input audio will be delayed by one loop before being mixed with model
    # output.
    beats_per_loop = ui_params["beats_per_loop"]
    loop_seconds = beats_per_loop * 60 / self.bpm
    loop_samples = int(loop_seconds * SAMPLE_RATE)

    # "I/O offset" is the number of samples by which we shift the inputs
    # before mixing them with the outputs.
    io_offset = CHUNK_SAMPLES - int(
        self.system.config.crossfade_length * SAMPLE_RATE)
    if not self.use_prerecorded_input:
      assert self.latency_samples is not None, ("latency_samples must be set in live mode.")
      io_offset += loop_samples - self.latency_samples
    assert io_offset >= 0, ("Increase `beats_per_loop` in the previous cell and rerun it.")

    # Select a window of input audio for mixing.
    inputs_to_mix = self.injection_state.all_inputs[ -(io_offset + mix_samples):-io_offset]

    # Select a window of output audio for mixing.
    outputs_to_mix = self.injection_state.all_outputs[-mix_samples:]
    outputs_to_mix *= ui_params.get("model_feedback", 0.0)

    # Silence the last `input_gap_ms` ms of `inputs_to_mix`, to discourage
    # copying the input verbatim.
    input_gap_samples = int(SAMPLE_RATE * ui_params.get("input_gap", 0.0) / 1000)
    ramp_samples = 100
    ramp = np.linspace(1, 0, min(ramp_samples, input_gap_samples))
    if self.use_prerecorded_input:
        ramp = np.stack([ramp, ramp], axis=-1)
    envelope = np.concatenate(
        [np.ones_like(inputs_to_mix[input_gap_samples:]),
         ramp,
         np.zeros_like(inputs_to_mix[:max(0, input_gap_samples - ramp_samples)])
        ]
    )
    inputs_to_mix = inputs_to_mix * envelope

    # Mix input and output audio.
    if not self.use_prerecorded_input:
      inputs_to_mix = inputs_to_mix[:, None]
    mix_audio = audio_lib.Waveform(
        inputs_to_mix + outputs_to_mix,
        sample_rate=SAMPLE_RATE
    )
    # Encode mix audio to tokens, and throw away a prefix.
    mix_tokens = self.spectrostream_model.encode(mix_audio)[
        LEFT_EDGE_FRAMES_TO_REMOVE:]

    if self.state is not None:
      self.injection_state.context_tokens_orig = self.state.context_tokens
      self.state.context_tokens[-len(mix_tokens):] = mix_tokens[
          :, :self.system.config.decoder_codec_rvq_depth]

    max_decode_frames = round(
        CHUNK_SECONDS * self.system.config.codec_frame_rate)

    chunk, self.state = self.system.generate_chunk(
        state=self.state,
        style=self.get_style_embedding(),
        seed=0,
        max_decode_frames=max_decode_frames,
        context_tokens_orig=self.injection_state.context_tokens_orig,
        **ui_params,
    )

    # Add this chunk (before cross-fading) to the end of `all_outputs`.
    # Note, we ignore the first frame of the chunk, which will be used for
    # cross-fading.
    self.injection_state.all_outputs = np.concatenate(
        [self.injection_state.all_outputs,
         chunk.samples[self.fade.fade_size:]]
    )

    chunk = self.fade(chunk.samples)
    chunk *= ui_params.get("model_volume", 1.0)

    if ui_params.get("metronome"):
      # Add metronome audio to output.
      start = (self.injection_state.step * CHUNK_SAMPLES) % loop_samples
      end = start + CHUNK_SAMPLES
      metronome_chunk = self.metronome_audio[start:end]
      chunk += metronome_chunk[:, None]

    if self.use_prerecorded_input:
      chunk += inputs * ui_params.get("input_volume")

    # When intro loops are over, raise model volume and feedback.
    if self.injection_state.step + 1 == int(self.intro_loops * loop_samples / CHUNK_SAMPLES):
      if self.ui_config is not None:
        self.ui_config.set("model_feedback", 0.95)
        self.ui_config.set(
            "model_volume",
            0.6 if self.use_prerecorded_input else 0.95,
        )
        if not self.use_prerecorded_input:
          self.ui_config.set("metronome", False)
          self.ui_config.set("input_gap", 400)

    self.injection_state.step += 1
    return chunk

  def __call__(self, inputs):
    return self.generate(self.global_ui_params(), inputs)
