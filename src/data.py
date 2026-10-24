"""Data loading, cleaning, vocabulary and batching for the Drugs.com review dataset.

Task: 3-class sentiment classification of patient reviews, following Gräßer et al. (2018):
    rating 1-4  -> negative (0)
    rating 5-6  -> neutral  (1)
    rating 7-10 -> positive (2)
"""
import html
import json
import os
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

LABELS = ["negative", "neutral", "positive"]
PAD, UNK = "<pad>", "<unk>"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?.,;:()]")


def rating_to_label(rating):
    if rating <= 4:
        return 0
    if rating <= 6:
        return 1
    return 2


def clean_text(text):
    text = html.unescape(text).strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return re.sub(r"\s+", " ", text).lower()


def tokenize(text):
    return TOKEN_RE.findall(text)


def load_clean_splits(raw_dir, val_frac=0.1, seed=42):
    """Read the official train/test files, remove duplicated reviews and leakage, build a validation split.

    The same review text is often published under several drug names (brand + generic), so ~30% of the
    training reviews are duplicates and ~60% of the official test reviews also appear in the training set.
    We keep one copy of each review and drop test reviews already seen in training.
    """
    train = pd.read_json(os.path.join(raw_dir, "train.jsonl"), lines=True)
    test = pd.read_json(os.path.join(raw_dir, "test.jsonl"), lines=True)
    stats = {"raw_train": len(train), "raw_test": len(test)}

    for df in (train, test):
        df["text"] = df["review"].map(clean_text)
        df["label"] = df["rating"].map(rating_to_label)

    train = train[train["text"].str.len() > 0]
    test = test[test["text"].str.len() > 0]
    train = train.drop_duplicates("text", keep="first")
    stats["dedup_train"] = len(train)
    test = test.drop_duplicates("text", keep="first")
    stats["dedup_test"] = len(test)
    test = test[~test["text"].isin(set(train["text"]))]
    stats["test_after_leakage_removal"] = len(test)

    train, val = train_test_split(train, test_size=val_frac, stratify=train["label"], random_state=seed)
    stats.update(train=len(train), val=len(val), test=len(test))
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}, stats


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, min_freq=3, max_size=30000):
        counts = Counter(tok for toks in token_lists for tok in toks)
        words = [w for w, c in counts.most_common() if c >= min_freq][: max_size - 2]
        return cls([PAD, UNK] + words)

    def encode(self, tokens):
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def __len__(self):
        return len(self.itos)


def prepare(raw_dir, cache_path, max_len=200, min_freq=3, max_vocab=30000, seed=42):
    """Tokenize and encode all splits once; cache the result to disk."""
    if os.path.exists(cache_path):
        return torch.load(cache_path)
    splits, stats = load_clean_splits(raw_dir, seed=seed)
    tokens = {name: df["text"].map(tokenize).tolist() for name, df in splits.items()}
    vocab = Vocab.build(tokens["train"], min_freq=min_freq, max_size=max_vocab)
    data = {"vocab": vocab.itos, "stats": stats, "max_len": max_len}
    for name, df in splits.items():
        data[name] = {
            "ids": [vocab.encode(t[:max_len]) or [vocab.stoi[UNK]] for t in tokens[name]],
            "labels": df["label"].tolist(),
            "full_lengths": [len(t) for t in tokens[name]],
        }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(data, cache_path)
    return data


def stratified_subset(labels, size, seed):
    """Indices of a class-stratified subset (keeps the original class proportions)."""
    if size is None or size >= len(labels):
        return list(range(len(labels)))
    idx, _ = train_test_split(np.arange(len(labels)), train_size=size, stratify=labels, random_state=seed)
    return sorted(idx.tolist())


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, split, indices=None):
        self.ids = split["ids"]
        self.labels = split["labels"]
        self.indices = list(range(len(self.ids))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return self.ids[j], self.labels[j]

    def lengths(self):
        return [len(self.ids[j]) for j in self.indices]


class BucketBatchSampler(torch.utils.data.Sampler):
    """Groups reviews of similar length to limit padding (and wasted recurrent steps).

    Indices are shuffled, cut into pools of `pool` batches, sorted by length inside each pool,
    then batches are shuffled. Without shuffling (evaluation), batches are fully sorted by length.
    """

    def __init__(self, lengths, batch_size, shuffle, pool=50, seed=0):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pool = pool
        self.rng = random.Random(seed)

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            self.rng.shuffle(idx)
            pool_size = self.batch_size * self.pool
            pools = [sorted(idx[i:i + pool_size], key=self.lengths.__getitem__)
                     for i in range(0, len(idx), pool_size)]
            batches = [p[i:i + self.batch_size] for p in pools for i in range(0, len(p), self.batch_size)]
            self.rng.shuffle(batches)
        else:
            idx.sort(key=self.lengths.__getitem__)
            batches = [idx[i:i + self.batch_size] for i in range(0, len(idx), self.batch_size)]
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    ids = torch.zeros(len(seqs), int(lengths.max()), dtype=torch.long)
    for k, s in enumerate(seqs):
        ids[k, : len(s)] = torch.tensor(s)
    return ids, lengths, torch.tensor(labels)


def make_loader(split, batch_size, shuffle, indices=None, seed=0):
    ds = ReviewDataset(split, indices)
    sampler = BucketBatchSampler(ds.lengths(), batch_size, shuffle, seed=seed)
    return torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = prepare(os.path.join(root, "data", "raw"), os.path.join(root, "data", "processed", "drugs.pt"))
    print(json.dumps(d["stats"], indent=2), "vocab", len(d["vocab"]))
