import numpy as np


class FakeEmbedder:
    def __init__(self, vectors):
        self.vectors = {
            text: np.asarray(vector, dtype=float)
            for text, vector in vectors.items()
        }

    def encode(self, texts):
        return np.asarray([
            self.vectors[text]
            for text in texts
        ])