
import numpy as np
import pandas as pd


def keep_n(text, X, y, n):
  counts = dict()
  new_text, new_X, new_y = list(), list(), list()
  for t_, x_, y_ in zip(text, X, y):
    if y_ not in counts:
      counts[y_] = 0

    if counts[y_] >= n: continue

    counts[y_] += 1

    new_text.append(t_)
    new_X.append(x_)
    new_y.append(y_)

  return new_text, np.array(new_X), np.array(new_y)
  

def count_labels(y):
  counts = dict()
  for y_ in y:
    if y_ not in counts:
      counts[y_] = 0
    counts[y_] += 1
  return counts


def _make_numeric_labels(labels, label_to_y=None):
  if label_to_y is not None:
    return ([label_to_y(label) for label in labels],
            {label_to_y(label): label for label in set(labels)})
  emoji_to_n = dict()
  numeric_labels = list()
  c = 0
  for emoji in labels:
    if emoji not in emoji_to_n:
      emoji_to_n[emoji] = c
      c += 1
    numeric_labels.append(emoji_to_n[emoji])
  return numeric_labels, {emoji_to_n[emoji]: emoji for emoji in emoji_to_n}


def prepare(text, emoji, representation, most_frequent=30, emoji_to_y=None):
  X, y = list(), list()
  filtered_text = list()
  counts = dict()
  for i, (t, e) in enumerate(zip(text, emoji)):
    x = representation(t)
    if x is None: continue

    filtered_text.append(t)
    X.append(x)
    y.append(e)

    if e not in counts:
      counts[e] = 0
    counts[e] += 1

    # if i % 1000 == 0:
    #   print(f'{i} / {len(text)}')

  emojis_to_keep = {
    e for c, e in sorted(
      [(counts[emoji], emoji) for emoji in counts], 
      reverse=True)[:most_frequent]}

  filtered_text, X, y = zip(
    *[(t, x, e) for t, x, e in zip(filtered_text, X, y) if e in emojis_to_keep])

  y, y_to_emoji = _make_numeric_labels(y, emoji_to_y)

  return filtered_text, np.array(X), np.array(y), y_to_emoji


def load(filename):
  return pd.read_csv(filename, sep='\t')
