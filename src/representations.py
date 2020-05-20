
import numpy as np


class AverageWordEmbeddingRepresentation:

  def __init__(self, tokenizer, embedding):
    self._tokenizer = tokenizer
    self._embedding = embedding

  def __call__(self, text):
    wes = []
    for token in self._tokenizer(text):
      we = self._embedding(token)
      if we is not None:
        wes.append(we)
    if len(wes) == 0:
      return None
    return np.mean(np.array(wes), axis=0)
