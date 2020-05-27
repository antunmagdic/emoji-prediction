import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, embs, num_layers=1, dropout=0., bidirectional=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.embs = embs

        self.rnn1 = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.fc1 = nn.Linear(hidden_size * self.num_directions, num_class)

    def forward(self, x, lens):
        batch_size = x.size(0)

        x = torch.transpose(self.embs(x), 0, 1)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lens, enforce_sorted=False)

        hidden = (
            torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size),
            torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        )
        
        _, (h, _) = self.rnn1(x, hidden)

        x = h.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        x = torch.cat(tuple(x[-1]), 1)

        x = self.fc1(x)

        return x

    def get_last_hidden(self, x, lens):
        with torch.no_grad():
            batch_size = x.size(0)

            x = torch.transpose(self.embs(x), 0, 1)
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lens, enforce_sorted=False)

            hidden = (
                torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size),
                torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size)
            )
            
            _, (h, _) = self.rnn1(x, hidden)

            return h

    def save(self):
        fname = f'{self.input_size}_{self.hidden_size}_{self.num_class}_{self.num_layers}_{self.num_directions}_.pt'
        torch.save(self.state_dict(), fname)

        return fname

    def load(path, embs, dropout):
        spl = path.split('/')
        params = spl[-1].split('_')

        input_size = int(params[0])
        hidden_size = int(params[1])
        num_class = int(params[2])
        num_layers = int(params[3])
        bidirectional = False if int(params[4]) == 1 else True

        model = RNN(input_size, hidden_size, num_class, embs, num_layers, dropout, bidirectional)
        model.load_state_dict(torch.load(path))
        model.eval()

        return model


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
