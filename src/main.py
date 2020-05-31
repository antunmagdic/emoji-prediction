"""
Usage:
python3 main.py glove.txt data.tsv
"""

import argparse
import data
import embeddings
import plot
import representations
import tokens

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

  # clf = LogisticRegression(multi_class='auto')
  clf = KNeighborsClassifier(10)

  print('Fitting the classifier...', end='', flush=True)
  clf.fit(X_train, y_train)
  print('\rFitting the classifier. Done.')

  print('Evaluating on train...', end='', flush=True)
  train_predicted = clf.predict(X_train)
  print('\rEvaluating on train. Done.')
  print('Evaluating on test...', end='', flush=True)
  test_predicted = clf.predict(X_test)
  print('\rEvaluating on test. Done.')
  print(f'Accuracy on train: {accuracy_score(y_train, train_predicted)}')
  print(f'Accuracy on test:  {accuracy_score(y_test, test_predicted)}')

  for k, v in sorted([(k, y_to_emoji[k]) for k in y_to_emoji]):
    print(f'{k}: {v}')
  plot.confusion_matrix(confusion_matrix(y_test, test_predicted), set(y_test))


if __name__ == '__main__':
  main()