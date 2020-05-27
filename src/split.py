"""
Usage:
python3 split.py data.tsv output_train.tsv output_valid.tsv output_test.tsv
"""

import argparse
import data
import embeddings
import plot
import representations
import tokens

from sklearn.model_selection import train_test_split


TRAIN_SIZE = 0.6
VALID_SIZE = 0.2
TEST_SIZE = 0.2


def write_to_file(filename, texts, emojis):
  with open(filename, 'w') as file:
    file.write('text\temoji\n')
    file.write(
      '\n'.join([f'{text}\t{emoji}' for text, emoji in zip(texts, emojis)]))
    file.write('\n')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('datafile')
  parser.add_argument('trainfile')
  parser.add_argument('validfile')
  parser.add_argument('testfile')
  args = parser.parse_args()

  print('Loading data...', end='', flush=True)
  dataset = data.load(args.datafile)
  print('\rLoading data. Done.')
  
  representation = lambda x: x
  text, X, y, y_to_emoji = data.prepare(dataset.text, dataset.emoji,
                                        representation)
  print('\rPreparing data. Done.')

  # text, X, y = data.keep_n(text, X, y, 8000)

  X_train, X_valid_test, y_train, y_valid_test = train_test_split(
    X, y, test_size=VALID_SIZE + TEST_SIZE, stratify=y)

  X_valid, X_test, y_valid, y_test = train_test_split(
    X_valid_test, y_valid_test, test_size=TEST_SIZE / (VALID_SIZE + TEST_SIZE),
    stratify=y_valid_test)

  write_to_file(args.trainfile, X_train, [y_to_emoji[y_] for y_ in y_train])
  write_to_file(args.validfile, X_valid, [y_to_emoji[y_] for y_ in y_valid])
  write_to_file(args.testfile, X_test, [y_to_emoji[y_] for y_ in y_test])


if __name__ == '__main__':
  main()