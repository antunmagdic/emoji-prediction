
import nltk

_tokenizer = nltk.tokenize.TweetTokenizer()

def tokenize_for_glove(tweet):
  for token in _tokenizer.tokenize(tweet):
    if '\'' in token:
      i = token.index('\'')
      pre = token[:i]
      post = token[i:]
      if len(pre.strip()) != 0: yield pre
      if len(post.strip()) != 0: yield post
    else:
      yield token
  