"""
Usage:
python3 kmeans.py glove.txt data.tsv
"""

import argparse
import data
import embeddings
import plot
import representations
import tokens

import numpy as np

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def emoji_counts(y, y_to_emoji):
  return {y_to_emoji[y_]: len([y__ for y__ in y if y__ == y_]) / len(y) for y_ in set(y)}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('word_embeddings')
  parser.add_argument('datafile')
  args = parser.parse_args()

  print('Loading data...', end='', flush=True)
  dataset = data.load(args.datafile)
  print('\rLoading data. Done.')

  print('Loading word embeddings...', end='', flush=True)
  embedding = embeddings.load_from_file(args.word_embeddings)
  print('\rLoading word embeddings. Done.')

  representation = representations.AverageWordEmbeddingRepresentation(
    tokens.tokenize_for_glove, embedding)
  
  print('Preparing data...', end='', flush=True)
  text, X, y, y_to_emoji = data.prepare(dataset.text, dataset.emoji,
                                        representation)
  print('\rPreparing data. Done.')

  # X, _, y, _ = train_test_split(X, y, test_size=0.8)

  n_clusters = 50
  cluster = KMeans(n_clusters=n_clusters)

  print('Clustering...', end='', flush=True)
  cluster.fit(X, y)
  print('\rClustering. Done.')

  clusters = np.array(
    [cluster.labels_ == k for k in range(n_clusters)])

  text = np.array(text)
  for i, cluster in enumerate(clusters):
    y_ = y[cluster]
    text_ = text[cluster]
    with open(f'cluster{i}.tsv', 'w') as file:
      file.write('\n'.join([f'{t_, y_to_emoji[label_]}' for t_, label_ in zip(text[cluster], y_)]))
    print()
    print(i, len(y_))
    print([(e, round(100 * c, 2)) for c, e in sorted(
      [(v, k) for k, v in emoji_counts(y_, y_to_emoji).items()], 
      reverse=True)])


if __name__ == '__main__':
  main()
