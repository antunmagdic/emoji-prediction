
import numpy as np


def _is_number(s):
  s = s[1:] if s[0] == '-' else s
  return s.replace('.', '', 1).isnumeric()


def _is_url(s):
  return '://' in s


class EmbeddingProvider:

  def __init__(self, embedding_dict, ignore_case=True):
    self._embeddings = embedding_dict
    self._ignore_case = ignore_case

  def __call__(self, word):
    if len(word) == 0: return None
    try:
      if self._ignore_case:
        return self._embeddings[word.lower()]
      return self._embeddings[word]
    except KeyError: pass
    
    try:
      if word[0] == '@':
        return self._embeddings['<user>']
      if word[0] == '#':
        return self._embeddings['<hashtag>']
      if _is_number(word):
        return self._embeddings['<number>']
      if _is_url(word): 
        return self._embeddings['<url>']
    except KeyError: pass
    
    return None


def load_from_file(filename, ignore_case=True):
  embeddings = dict()
  with open(filename) as file:
    for line in file.readlines():
      line = line[:-1].split(' ')
      word = line[0]
      if ignore_case:
        word = word.lower()
      vector = np.array([float(x) for x in line[1:]])
      embeddings[word] = vector
  return EmbeddingProvider(embeddings)
