
import argparse
import data
import embeddings
import representations
import tokens

from sklearn.linear_model import LogisticRegression

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
                                        representation, most_frequent=20)
  print('\rPreparing data. Done.')

  text, X, y = data.keep_n(text, X, y, min(data.count_labels(y).values()))

  with open('data/clean_data30.tsv', 'w') as f:
    f.write('text\temoji\n')
    f.write('\n'.join([f'{t_}\t{y_to_emoji[y_]}' for t_, y_ in zip(text, y)]))


if __name__ == '__main__':
  main()