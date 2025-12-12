import dataclasses
import numpy as np


@dataclasses.dataclass
class AudioInjectionState:
  """State management for Audio Injection."""
  # The most recent context window (10s) of audio tokens corresponding to the
  # model's predicted output. These are parallel to `state.context_tokens` but
  # that context has the input audio mixed in.
  context_tokens_orig: np.ndarray
  # Stores all audio input (mono for live input, stereo for prerecorded input)
  all_inputs: np.ndarray
  # Stores all audio output (stereo)
  all_outputs: np.ndarray
  # How many chunks of audio have been generated
  step: int

