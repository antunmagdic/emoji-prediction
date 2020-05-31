"""
Usage:
python3 mlp_glove.py glove.txt train_data.tsv valid_data.tsv test_data.tsv
"""

EMOJI_TO_Y = {
  '❤': 0,
  '💕': 1,
  '💜': 2,
  '💙': 3,
  '🖤': 4,
  '😍': 5,
  '😊': 6,
  '😭': 7,
  '😔': 8,
  '😩': 9,
  '😂': 10,
  '🤣': 11,
  '🤔': 12,
  '🙄': 13,
  '👀': 14,
  '✨': 15,
  '💯': 16,
  '🔥': 17,
  '💀': 18,
  '🎄': 19
}

import argparse
import data
import embeddings
import plot
import representations
import tokens

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('word_embeddings')
  parser.add_argument('train')
  parser.add_argument('valid')
  parser.add_argument('test')
  args = parser.parse_args()

  print('Loading data...', end='', flush=True)
  dataset_train = data.load(args.train)
  dataset_valid = data.load(args.valid)
  dataset_test = data.load(args.test)
  print('\rLoading data. Done.')

  print('Loading word embeddings...', end='', flush=True)
  embedding = embeddings.load_from_file(args.word_embeddings)
  print('\rLoading word embeddings. Done.')

  representation = representations.AverageWordEmbeddingRepresentation(
    tokens.tokenize_for_glove, embedding)
  
  print('Preparing data...', end='', flush=True)
  text_train, X_train, y_train, y_to_emoji = data.prepare(
    dataset_train.text, dataset_train.emoji, representation,
    emoji_to_y=lambda emoji: EMOJI_TO_Y[emoji])
  text_valid, X_valid, y_valid, y_to_emoji = data.prepare(
    dataset_valid.text, dataset_valid.emoji, representation,
    emoji_to_y=lambda emoji: EMOJI_TO_Y[emoji])
  text_test, X_test, y_test, y_to_emoji = data.prepare(
    dataset_test.text, dataset_test.emoji, representation,
    emoji_to_y=lambda emoji: EMOJI_TO_Y[emoji])
  print('\rPreparing data. Done.')

  clf = MLPClassifier(hidden_layer_sizes=[150, 100, 50], max_iter=500, alpha=0.0001)

  print('Fitting the classifier...', end='', flush=True)
  clf.fit(X_train, y_train)
  print('\rFitting the classifier. Done.')

  print('Evaluating on train...', end='', flush=True)
  train_predicted = clf.predict(X_train)
  print('\rEvaluating on train. Done.')
  print('Evaluating on valid...', end='', flush=True)
  valid_predicted = clf.predict(X_valid)
  print('\rEvaluating on valid. Done.')
  print('Evaluating on test...', end='', flush=True)
  test_predicted = clf.predict(X_test)
  print('\rEvaluating on test. Done.')
  print(f'Accuracy on train: {accuracy_score(y_train, train_predicted)}')
  print(f'Accuracy on valid: {accuracy_score(y_valid, valid_predicted)}')
  print(f'Accuracy on test:  {accuracy_score(y_test, test_predicted)}')

  for k, v in sorted([(k, y_to_emoji[k]) for k in y_to_emoji]):
    print(f'{k}: {v}')
  plot.confusion_matrix(confusion_matrix(y_test, test_predicted), set(y_test))


if __name__ == '__main__':
  main()