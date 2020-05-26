# run example
# python3 src/trainer.py data/glove.twitter.27B/glove.twitter.27B.100d.txt data/clean_data20.tsv

import argparse
import torch
import time

import vocab
import network


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('word_embeddings')
  parser.add_argument('datafile')
  args = parser.parse_args()

  test_part = .2
  train_batch_size = 128
  test_batch_size = 128
  embedding_size = 100
  hidden_size = 300
  classes = 20
  epochs = 20
  learning_rate = 1e-3

  print('Loading data...', end='', flush=True)
  train_dataset, test_dataset, voc = vocab.load_dataset(args.datafile, test_part, train_batch_size, test_batch_size)
  print('\rLoading data. Done.')

  print('Loading word embeddings...', end='', flush=True)
  embs = vocab.get_pretrained_embeddings(voc, args.word_embeddings, embedding_size)
  print('\rLoading word embeddings. Done.')

  model = network.RNN(embedding_size, hidden_size, classes, embs)

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  for epoch in range(epochs):
    start_time = time.time()
    train_loss, train_acc = network.train(model, train_dataset, optimizer, criterion)
    valid_loss, valid_acc = network.evaluate(model, test_dataset, criterion)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')


if __name__ == '__main__':
  main()
