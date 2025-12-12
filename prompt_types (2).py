
class TextPrompt(str):
    """ text prompt."""


class AudioPrompt:
    """ raw audio samples (list/np array)."""
    def __init__(self, value):
        self.value = value   


class EmbeddingPrompt:
    """ Embedding vector."""
    def __init__(self, value):
        self.value = value   
