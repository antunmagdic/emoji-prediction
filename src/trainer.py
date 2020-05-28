# run example
# python3 src/trainer.py data/glove.twitter.27B/glove.twitter.27B.100d.txt data/clean_data20.tsv

import argparse
import torch
import time

import vocab
import network


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('train_datafile')
  parser.add_argument('valid_datafile')
  parser.add_argument('test_datafile')
  parser.add_argument('word_embeddings')
  args = parser.parse_args()

  train_batch_size = 128
  valid_batch_size = 128
  test_batch_size = 128
  embedding_size = 2
  hidden_size = 2
  classes = 20
  num_layers = 1
  dropout = 0
  bidirectional = False
  epochs = 1
  learning_rate = 1e-3
  save = False

  print('Loading data...', end='', flush=True)
  train_dataset, valid_dataset, test_dataset, voc = vocab.load_dataset(args.train_datafile, args.valid_datafile, args.test_datafile, train_batch_size, valid_batch_size, test_batch_size)
  print('\rLoading data. Done.')

  print('Loading word embeddings...', end='', flush=True)
  embs = vocab.get_random_embeddings(voc, embedding_size)
  #embs = vocab.get_pretrained_embeddings(voc, args.word_embeddings, embedding_size)
  print('\rLoading word embeddings. Done.')

  model = network.RNN(embedding_size, hidden_size, classes, embs, num_layers, dropout, bidirectional)

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  print('Training...\n')

  for epoch in range(epochs):
    start_time = time.time()
    train_loss, train_acc = network.train(model, train_dataset, optimizer, criterion)
    valid_loss, valid_acc = network.evaluate(model, valid_dataset, criterion)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

  test_loss, test_acc = network.evaluate(model, test_dataset, criterion)
  print('\nTest evaluation...')
  print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')

  print('\nTraining done.')

  if save:
    print('Saving model...', end='', flush=True)
    fname = model.save()
    print('\rSaving model. Done.')
    print('Model saved in', fname)


if __name__ == '__main__':
  main()
