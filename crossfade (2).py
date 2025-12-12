import numpy as np


class AudioFade:

  def __init__(self, chunk_size: int, num_chunks: int, stereo: bool):
    fade_size = chunk_size * num_chunks
    self.fade_size = fade_size
    self.num_chunks = num_chunks

    self.previous_chunk = np.zeros(fade_size)
    self.ramp = np.sin(np.linspace(0, np.pi / 2, fade_size)) ** 2

    if stereo:
      self.previous_chunk = self.previous_chunk[:, np.newaxis]
      self.ramp = self.ramp[:, np.newaxis]

  def reset(self):
    self.previous_chunk = np.zeros_like(self.previous_chunk)

  def __call__(self, chunk: np.ndarray) -> np.ndarray:
    chunk[: self.fade_size] *= self.ramp
    chunk[: self.fade_size] += self.previous_chunk
    self.previous_chunk = chunk[-self.fade_size :] * np.flip(self.ramp)
    return chunk[: -self.fade_size]
