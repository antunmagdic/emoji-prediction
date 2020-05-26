import random
import torch
import pandas as pd

import tokens


CHAR_PAD = '<pad>'
CHAR_UNK = '<unk>'


class Instance:
    def __init__(self, words, label):
        self.words = words
        self.label = label
    
    def __iter__(self):
        yield self.words
        yield self.label


class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, instances):
        self.instances = instances
        self.text_vocab = None
        self.label_vocab = None
    
    def create_vocab(self):
        frequencies_text = {}
        frequencies_label = {}
        
        for words, label in self.instances:
            for w in words:
                if w not in frequencies_text:
                    frequencies_text[w] = 1
                else:
                    frequencies_text[w] += 1

            if label not in frequencies_label:
                frequencies_label[label] = 1
            else:
                frequencies_label[label] += 1
        
        return Vocab(frequencies_text), Vocab(frequencies_label, specials=False)
    
    def set_vocab(self, text_vocab, label_vocab):
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab

    def load(text, emoji, tokenizer, ignore_case=True):
        instances = []
        
        for i, (t, e) in enumerate(zip(text, emoji)):
            words = []

            for token in tokenizer(t):
                if token[0] == '@':
                    words += ['<user>']
                elif token[0] == '#':
                    words += ['<hashtag>']
                elif _is_number(token):
                    words += ['<number>']
                elif _is_url(token): 
                    words += ['<url>']
                else:
                    words += [token.lower() if ignore_case else token]

            instances += [Instance(words, e)]
        
        return NLPDataset(instances)

    def __getitem__(self, i):
        return (
            self.text_vocab.encode(self.instances[i].words), 
            self.label_vocab.encode(self.instances[i].label)
        )

    def __len__(self):
        return len(self.instances)


class Vocab:
    def __init__(self, frequencies, max_size=-1, min_freq=0, specials=True):
        self.stoi = {}
        self.itos = {}

        if max_size == -1:
            max_size = len(frequencies) + 2

        cnt = 0

        if specials:
            if max_size >= 1:
                self.stoi[CHAR_PAD] = 0
                self.itos[0] = CHAR_PAD
                cnt += 1

            if max_size >= 2:
                self.stoi[CHAR_UNK] = 1
                self.itos[1] = CHAR_UNK
                cnt += 1

        for k, v in sorted(frequencies.items(), key=lambda x: x[1], reverse=True):
            if cnt > max_size:
                break

            if v < min_freq:
                break

            self.stoi[k] = cnt
            self.itos[cnt] = k

            cnt += 1

    def encode(self, instance):
        if isinstance(instance, str):
            return torch.tensor(self.stoi[instance])
        else:
            return torch.tensor([self.stoi[s] if s in self.stoi else self.stoi[CHAR_UNK] for s in instance])

    def __len__(self):
        return len(self.stoi)


def _is_number(s):
  s = s[1:] if s[0] == '-' else s
  return s.replace('.', '', 1).isnumeric()


def _is_url(s):
  return '://' in s


def get_random_embeddings(vocab, size):
    emb = torch.randn(len(vocab), size)
    emb[0, :] = 0.

    return torch.nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=0)


def get_pretrained_embeddings(vocab, file_name, size, ignore_case=True):
    emb = torch.randn(len(vocab), size)
    emb[0, :] = 0.

    with open(file_name, 'r') as f:
        for line in f:
            l = line.strip().split()

            word = l[0].lower() if ignore_case else l[0]

            if word in vocab.stoi:
                emb[vocab.stoi[word]] = torch.tensor([float(i) for i in l[1:]])

    return torch.nn.Embedding.from_pretrained(emb, freeze=True, padding_idx=0)


def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)

    lengths = torch.tensor([len(text) for text in texts])

    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_index)
    labels = torch.tensor(labels)

    return texts, labels, lengths


def train_test_split(dataset, test_part):
    random.shuffle(dataset.instances)
    shuffled = dataset.instances

    N_test = int(test_part * len(shuffled))

    train_dataset = NLPDataset(shuffled[N_test:])
    test_dataset = NLPDataset(shuffled[:N_test])

    return train_dataset, test_dataset


def load_dataset(file_name, test_part, train_batch, test_batch):
    dataset_pd = load(file_name)
    dataset = NLPDataset.load(dataset_pd.text, dataset_pd.emoji, tokens.tokenize_for_glove)

    text_vocab, label_vocab = dataset.create_vocab()

    train_dataset, test_dataset = train_test_split(dataset, test_part)

    train_dataset.set_vocab(text_vocab, label_vocab)
    test_dataset.set_vocab(text_vocab, label_vocab)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch, shuffle=True, collate_fn=pad_collate_fn)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch, shuffle=False, collate_fn=pad_collate_fn)

    return train_data_loader, test_data_loader, text_vocab


def load(filename):
  return pd.read_csv(filename, sep='\t')
