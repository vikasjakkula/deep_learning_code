# data/

Drop real datasets here to override the built-in synthetic fallbacks.

## Seeds (`seeds_dataset.csv`)
UCI "seeds" wheat dataset — 210 rows, 7 numeric features, last column = class
(1/2/3, automatically remapped to 0/1/2). No header.
Source: https://archive.ics.uci.edu/dataset/236/seeds
The original `.txt` is tab/space separated; convert it to comma-separated
`seeds_dataset.csv` (7 features + label per line).

## Heart disease (`heart.csv`)
13 features + binary target as the last column. With a header row.
Source: https://archive.ics.uci.edu/dataset/45/heart+disease

## Format expected by `nnscratch.data.load_csv`
```
f1,f2,...,fn,label_int
```
The last column must be the integer class label.

If a file is **absent**, `data.load_seeds()` / `load_heart()` generate a
deterministic synthetic dataset of the same shape (Gaussian blobs) so every
script runs fully offline.
