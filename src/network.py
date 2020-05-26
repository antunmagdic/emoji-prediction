import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, embs, num_layers=1, dropout=0., bidirectional=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.embs = embs

        self.rnn1 = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.fc1 = nn.Linear(hidden_size, num_class)

    def forward(self, x, lens):
        batch_size = x.size(0)

        x = torch.transpose(self.embs(x), 0, 1)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lens, enforce_sorted=False)

        h0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        c0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        
        _, (h, _) = self.rnn1(x, (h0, c0))
        
        x = self.fc1(h[-1])

        return x


def train(model, data, optimizer, criterion):
  model.train()

  N = 0
  train_loss = 0.
  train_acc = 0

  for batch_num, batch in enumerate(data):
    X, Y_, lens = batch
    batch_size = X.size(0)
    N += batch_size

    optimizer.zero_grad()

    logits = model(X, lens)

    loss = criterion(logits, Y_)
    train_loss += batch_size * loss.item()
    loss.backward()

    optimizer.step()

    train_acc += (logits.argmax(1) == Y_).sum().item()

  return train_loss / N, train_acc / N


def evaluate(model, data, criterion):
  model.eval()
  
  N = 0
  loss = 0.
  acc = 0
  
  with torch.no_grad():
    for batch_num, batch in enumerate(data):
      X, Y_, lens = batch
      batch_size = X.size(0)
      N += batch_size

      logits = model(X, lens)

      loss = criterion(logits, Y_)
      loss += batch_size * loss.item()

      acc += (logits.argmax(1) == Y_).sum().item()

  return loss / N, acc / N
