"""
Counts emoji labels in tsv files, reads from stdin. Usage:
python3 emoji_counts.py [limit] < {src.tsv}
Prints limit most frequent emojis, along their frequencies.
"""

import pandas as pd
import sys

counts = dict()
total = 0

for emoji in pd.read_csv(sys.stdin, sep='\t').emoji:
  if emoji not in counts:
    counts[emoji] = 0
  counts[emoji] += 1
  total += 1

counts = sorted([(counts[emoji], emoji) for emoji in counts], reverse=True)

limit = len(counts)
if len(sys.argv) >= 2:
  try:
    limit = int(sys.argv[1])
  except ValueError: pass

for i in range(limit):
  print('{:4d}. {:7d} {:10.5f}%   {}'.format(
    i + 1, counts[i][0], 100 * counts[i][0] / total, counts[i][1]))
