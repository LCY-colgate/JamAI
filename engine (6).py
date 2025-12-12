# @title Run this cell to initialize model (~5 minutes)

import os

import concurrent.futures
import dataclasses
import functools
import jax
import librosa
import numpy as np
import tempfile
from typing import Optional, Sequence, Tuple
import warnings

from magenta_rt import audio as audio_lib
from magenta_rt import musiccoca
from magenta_rt import spectrostream
from magenta_rt import system
from magenta_rt import utils


from constant import SAMPLE_RATE, CHUNK_SECONDS, CHUNK_SAMPLES

# ========= Helper functions ===========

def load_audio(audio_filename, sample_rate):
  """Loads an audio file.

  Args:
    audio_filename: File path to load.
    sample_rate: The number of samples per second at which the audio will be
        returned. Resampling will be performed if necessary.

  Returns:
    A numpy array of audio samples, sampled at the specified rate, in float32
    format.
  """
  y, unused_sr = librosa.load(audio_filename, sr=sample_rate, mono=False)
  return y


def wav_data_to_samples_librosa(audio_file, sample_rate):
  """Loads an in-memory audio file with librosa.

  Use this instead of wav_data_to_samples if the wav is 24-bit, as that's
  incompatible with wav_data_to_samples internal scipy call.

  Will copy to a local temp file before loading so that librosa can read a file
  path. Librosa does not currently read in-memory files.

  It will be treated as a .wav file.

  Args:
    audio_file: Wav file to load.
    sample_rate: The number of samples per second at which the audio will be
        returned. Resampling will be performed if necessary.

  Returns:
    A numpy array of audio samples, single-channel (mono) and sampled at the
    specified rate, in float32 format.
  """
  with tempfile.NamedTemporaryFile(suffix='.wav') as wav_input_file:
    wav_input_file.write(audio_file)
    # Before copying the file, flush any contents
    wav_input_file.flush()
    # And back the file position to top (not need for Copy but for certainty)
    wav_input_file.seek(0)
    return load_audio(wav_input_file.name, sample_rate)


def get_metronome_audio(
    loop_samples: int,
    beats_per_loop: int,
    sample_rate: int,
    chunk_samples: int):
  """Generates metronome audio.

  Args:
    loop_samples: The number of samples in a loop.
    beats_per_loop: The number of beats in a loop.
    sample_rate: The sample rate of the audio.
    chunk_samples: The number of samples in a chunk.

  Returns:
    A numpy array of metronome audio samples.
  """
  metronome_audio = np.zeros((loop_samples,))
  BEEP_SECONDS = 0.04
  BEEP_VOLUME = 0.25
  beeps = []
  for freq in (880, 440):
    beeps.append(BEEP_VOLUME * np.sin(np.linspace(
        0,
        2 * np.pi * freq * BEEP_SECONDS,
        int(BEEP_SECONDS * sample_rate))))
  ramp_samples = 100
  beep_envelope = np.concatenate(
      [np.linspace(0, 1, ramp_samples),
       np.ones((int(BEEP_SECONDS * sample_rate) - 2 * ramp_samples,)),
       np.linspace(1, 0, ramp_samples)])
  for i in range(len(beeps)):
    beeps[i] *= beep_envelope
  beat_length = loop_samples // beats_per_loop
  for i in range(beats_per_loop):
    beep = beeps[0 if i == 0 else 1]
    metronome_audio[i * beat_length:i * beat_length + len(beep)] = beep
  # Add an extra buffer to the metronome audio to make slicing easier later.
  return np.concat([metronome_audio, metronome_audio[:chunk_samples]])


# ================ Model System ======================
class MagentaRTCFGTied(system.MagentaRTT5X):
  """Magenta RT T5X system with "tied" CFG controlling input and style."""

  # This method is mostly identical to system.MagentaRTT5X.generate_chunk, but
  # adds "tied" CFG that acts jointly on input and style. Negative input tokens
  # are passed as `context_tokens_orig`.
  def generate_chunk(
      self,
      state: Optional[system.MagentaRTState] = None,
      style: Optional[musiccoca.StyleEmbedding] = None,
      seed: Optional[int] = None,
      **kwargs,
  ) -> Tuple[audio_lib.Waveform, system.MagentaRTState]:
    """Generates a chunk of audio and returns updated state.

    Args:
      state: The current state of the system.
      style: The style embedding to use for the generation.
      seed: The seed to use for the generation.
      **kwargs: Additional keyword arguments for sampling params, e.g.
        temperature, topk, guidance_weight, max_decode_frames.

    Returns:
      A tuple of the generated audio and the updated state.
    """
    # Init state, style, and seed (if not provided)
    if state is None:
      state = self.init_state()
    if seed is None:
      seed = np.random.randint(0, 2**31)

    context_tokens = {
        "orig": kwargs.get("context_tokens_orig", state.context_tokens),
        "mix": state.context_tokens,
    }
    codec_tokens_lm = {}
    for key, tokens in context_tokens.items():
      # Prepare codec tokens for LLM
      codec_tokens_lm[key] = np.where(
          tokens >= 0,
          utils.rvq_to_llm(
              np.maximum(tokens, 0),
              self.config.codec_rvq_codebook_size,
              self.config.vocab_codec_offset,
          ),
          np.full_like(tokens, self.config.vocab_mask_token),
      )
      assert (
          codec_tokens_lm[key].shape == self.config.context_tokens_shape
      )  # (250, 16)
      assert (
          codec_tokens_lm[key].min() >= self.config.vocab_mask_token
          and codec_tokens_lm[key].max()
          < (self.config.vocab_codec_offset + self.config.vocab_codec_size)
      )  # check range [1, 16386)

    # Prepare style tokens for LLM
    if style is None:
      style_tokens_lm = np.full(
          (self.config.encoder_style_rvq_depth,),
          self.config.vocab_mask_token,
          dtype=np.int32,
      )
    else:
      if style.shape != (self.style_model.config.embedding_dim,):
        raise ValueError(f"Invalid style shape: {style.shape}")
      style_tokens = self.style_model.tokenize(style)
      assert style_tokens.shape == (self.style_model.config.rvq_depth,)
      style_tokens = style_tokens[: self.config.encoder_style_rvq_depth]
      style_tokens_lm = utils.rvq_to_llm(
          style_tokens,
          self.config.style_rvq_codebook_size,
          self.config.vocab_style_offset,
      )
      assert (
          style_tokens_lm.min() >= self.config.vocab_style_offset
          and style_tokens_lm.max()
          < (self.config.vocab_style_offset + self.config.vocab_style_size)
      )  # check range [17140, 23554)
    assert style_tokens_lm.shape == (
        self.config.encoder_style_rvq_depth,
    )  # (6,)

    # Prepare encoder input
    batch_size, _, _ = self._device_params
    encoder_inputs_pos = np.concatenate(
        [codec_tokens_lm["mix"][
            :, :self.config.encoder_codec_rvq_depth].reshape(-1),
         style_tokens_lm
        ],
        axis=0,
    )
    assert encoder_inputs_pos.shape == (1006,)

    # Construct negative using original context tokens, and masking style.
    encoder_inputs_neg = np.concatenate(
        [codec_tokens_lm["orig"][
            :, :self.config.encoder_codec_rvq_depth].reshape(-1),
         style_tokens_lm
        ],
        axis=0,
    )
    encoder_inputs_neg[-self.config.encoder_style_rvq_depth:] = (
        self.config.vocab_mask_token)
    assert encoder_inputs_neg.shape == (1006,)

    encoder_inputs = np.stack([encoder_inputs_pos, encoder_inputs_neg], axis=0)
    assert encoder_inputs.shape == (2, 1006)

    # Generate tokens / NLL scores.
    max_decode_frames = kwargs.get(
        "max_decode_frames", self.config.chunk_length_frames
    )
    generated_tokens, _ = self._llm(
        {
            "encoder_input_tokens": encoder_inputs,
            "decoder_input_tokens": np.zeros(
                (
                    batch_size,
                    self.config.chunk_length_frames
                    * self.config.decoder_codec_rvq_depth,
                ),
                dtype=np.int32,
            ),
        },
        {
            "max_decode_steps": np.array(
                max_decode_frames * self.config.decoder_codec_rvq_depth,
                dtype=np.int32,
            ),
            "guidance_weight": kwargs.get(
                "guidance_weight", self._guidance_weight
            ),
            "temperature": kwargs.get("temperature", self._temperature),
            "topk": kwargs.get("topk", self._topk),
        },
        jax.random.PRNGKey(seed + state.chunk_index),
    )

    # Process generated tokens
    generated_tokens = np.array(generated_tokens)
    assert generated_tokens.shape == (
        batch_size,
        self.config.chunk_length_frames * self.config.decoder_codec_rvq_depth,
    )
    generated_tokens = generated_tokens[:1]  # larger batch sizes unsupported
    generated_tokens = generated_tokens.reshape(
        self.config.chunk_length_frames, self.config.decoder_codec_rvq_depth
    )  # (50, 16)
    generated_tokens = generated_tokens[:max_decode_frames]  # (N, 16)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      generated_rvq_tokens = utils.llm_to_rvq(
          generated_tokens,
          self.config.codec_rvq_codebook_size,
          self.config.vocab_codec_offset,
          safe=False,
      )

    # Decode via SpectroStream using additional frame of samples for crossfading
    # We want to generate a 2s chunk with an additional 40ms of crossfade, which
    # is one additional codec frame. Caller is responsible for actually applying
    # the crossfade.
    xfade_frames = state.context_tokens[-self.config.crossfade_length_frames :]
    if state.chunk_index == 0:
      # NOTE: This will create 40ms of gibberish at the beginning but it's OK.
      xfade_frames = np.zeros_like(xfade_frames)
    assert xfade_frames.min() >= 0
    xfade_tokens = np.concatenate([xfade_frames, generated_rvq_tokens], axis=0)
    assert xfade_tokens.shape == (
        self.config.crossfade_length_frames + max_decode_frames,
        self.config.decoder_codec_rvq_depth,
    )  # (N+1, 16)
    waveform = self.codec.decode(xfade_tokens)
    assert isinstance(waveform, audio_lib.Waveform)
    assert waveform.samples.shape == (
        self.config.crossfade_length_samples
        + max_decode_frames * self.config.frame_length_samples,
        self.num_channels,
    )  # ((N+1)*1920, 2)

    # Update state
    state.update(generated_rvq_tokens)

    return (waveform, state)


# Global variables to hold models (initially None)
_spectrostream_model = None
_magenta_rt_model = None


def load_models(verbose=True):
    """
    Explicitly load and initialize models.
    
    This gives you control over when the ~5 minute loading happens.
    
    Returns:
        tuple: (spectrostream_model, magenta_rt_model)
    """
    global _spectrostream_model, _magenta_rt_model
    
    if _spectrostream_model is not None and _magenta_rt_model is not None:
        if verbose:
            print("Models already loaded!")
        return _spectrostream_model, _magenta_rt_model
    
    if verbose:
        print("Loading models (this may take ~5 minutes)...")
        print("  1. Loading SpectroStream...")
    
    _spectrostream_model = spectrostream.SpectroStreamJAX(lazy=True)
    
    if verbose:
        print("  2. Loading Magenta RT (CFG Tied)...")
    
    _magenta_rt_model = MagentaRTCFGTied(tag="base", lazy=True)
    
    if verbose:
        print("✓ Models loaded successfully!")
    
    return _spectrostream_model, _magenta_rt_model

def get_spectrostream_model():
    """
    Get the SpectroStream encoder.
    """
    return _spectrostream_model


def get_magenta_rt_model():
    return _magenta_rt_model


def is_loaded():
    """Check if models are loaded."""
    return _magenta_rt_model is not None

