"""
Usage:
python3 naive_bayes.py data.tsv
"""

import argparse
import data
import embeddings
import plot
import representations
import tokens

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('datafile')
  args = parser.parse_args()

  dataset = data.load(args.datafile)
  representation = lambda x: x
  text, X, y, y_to_emoji = data.prepare(dataset.text, dataset.emoji,
                                        representation)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

  vectorizer = TfidfVectorizer()
  dim_reduction = TruncatedSVD(n_components=100)
  X_train = vectorizer.fit_transform(X_train)
  X_train = dim_reduction.fit_transform(X_train)
  X_test = vectorizer.transform(X_test)
  X_test = dim_reduction.fit_transform(X_test)
  
  clf = MultinomialNB(alpha=0)
  # clf = LogisticRegression(C=0.5)

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
