"""Exploratory data analysis: statistics (results/eda_stats.json) and figures (figures/eda_*.png)."""
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from data import clean_text, load_clean_splits, prepare, rating_to_label, tokenize  # noqa: E402
from plotstyle import CLASS_COLORS, CLASS_NAMES, INK2, MUTED, plt, save, setup  # noqa: E402

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
setup()

raw_train = pd.read_json(os.path.join(ROOT, "data", "raw", "train.jsonl"), lines=True)
raw_test = pd.read_json(os.path.join(ROOT, "data", "raw", "test.jsonl"), lines=True)
splits, split_stats = load_clean_splits(os.path.join(ROOT, "data", "raw"))
data = prepare(os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "data", "processed", "drugs.pt"))
train = splits["train"]
stats = {"splits": split_stats}

# --- duplicates: are duplicated texts consistent in rating? which drugs share them?
raw_train["text"] = raw_train["review"].map(clean_text)
dup_groups = raw_train[raw_train["text"].duplicated(keep=False)].groupby("text")
stats["dup_groups"] = int(dup_groups.ngroups)
stats["dup_groups_same_rating"] = float((dup_groups["rating"].nunique() == 1).mean())
stats["dup_groups_same_condition"] = float((dup_groups["condition"].nunique() == 1).mean())
stats["dup_groups_multi_drug"] = float((dup_groups["drugName"].nunique() > 1).mean())

# --- labels
for name, df in splits.items():
    stats[f"class_dist_{name}"] = (df["label"].value_counts(normalize=True).sort_index().round(4).tolist())
stats["rating_counts_train"] = train["rating"].value_counts().sort_index().astype(int).to_dict()

# --- lengths
lens = np.array(data["train"]["full_lengths"])
stats["len_train"] = {"mean": float(lens.mean()), "median": float(np.median(lens)), "p95": float(np.percentile(lens, 95)),
                      "p99": float(np.percentile(lens, 99)), "max": int(lens.max()),
                      "truncated_frac": float((lens > data["max_len"]).mean())}
lab = np.array(data["train"]["labels"])
stats["len_by_class_median"] = [float(np.median(lens[lab == k])) for k in range(3)]

# --- vocabulary
vocab = data["vocab"]
unk = 1
def oov_rate(split):
    ids = [i for seq in data[split]["ids"] for i in seq]
    return float(np.mean(np.array(ids) == unk))
stats["vocab_size"] = len(vocab)
stats["oov_rate"] = {s: oov_rate(s) for s in ("train", "val", "test")}
stats["n_drugs"] = int(train["drugName"].nunique())
stats["n_conditions"] = int(train["condition"].nunique())
stats["top_conditions"] = train["condition"].value_counts().head(10).astype(int).to_dict()

# --- most class-distinctive words (log-odds with informative Dirichlet prior, Monroe et al. 2008)
counts = [Counter(), Counter(), Counter()]
for text, y in zip(train["text"], train["label"]):
    counts[y].update(set(tokenize(text)))
prior = Counter()
for c in counts:
    prior.update(c)
a0 = sum(prior.values())
def log_odds(k, others):
    ck, co = counts[k], Counter()
    for o in others:
        co.update(counts[o])
    nk, no = sum(ck.values()), sum(co.values())
    out = {}
    for w, aw in prior.items():
        if aw < 50:
            continue
        aw = aw * 0.01
        l1 = np.log((ck[w] + aw) / (nk + a0 * 0.01 - ck[w] - aw))
        l2 = np.log((co[w] + aw) / (no + a0 * 0.01 - co[w] - aw))
        out[w] = (l1 - l2) / np.sqrt(1 / (ck[w] + aw) + 1 / (co[w] + aw))
    return [w for w, _ in sorted(out.items(), key=lambda t: -t[1])[:12]]
stats["distinctive_words"] = {CLASS_NAMES[k]: log_odds(k, [o for o in range(3) if o != k]) for k in range(3)}

with open(os.path.join(ROOT, "results", "eda_stats.json"), "w") as f:
    json.dump(stats, f, indent=1)
print(json.dumps(stats, indent=1))

# ---------------- figures
# 1) rating histogram coloured by sentiment class
fig, ax = plt.subplots(figsize=(5.2, 3.0))
rc = train["rating"].value_counts().sort_index()
colors = [CLASS_COLORS[rating_to_label(r)] for r in rc.index]
ax.bar(rc.index, rc.values / 1000, color=colors, width=0.8, edgecolor="#ffffff", linewidth=1)
ax.set_xticks(range(1, 11))
ax.set_xlabel("Rating (1-10)")
ax.set_ylabel("Reviews (thousands)")
ax.set_title("Ratings in the training set and the 3 sentiment classes")
dist = stats["class_dist_train"]
for k, (x, name) in enumerate(zip([2.5, 5.5, 8.5], CLASS_NAMES)):
    ax.text(x, ax.get_ylim()[1] * 0.92, f"{name}\n{dist[k]:.1%}", ha="center", va="top", color=INK2, fontsize=9)
ax.grid(axis="x", visible=False)
save(fig, os.path.join(FIG, "eda_ratings.png"))

# 2) review length distribution by class
fig, ax = plt.subplots(figsize=(5.2, 3.0))
bins = np.arange(0, 260, 8)
for k in range(3):
    ax.hist(np.clip(lens[lab == k], 0, 255), bins=bins, density=True, histtype="step", linewidth=2,
            color=CLASS_COLORS[k], label=f"{CLASS_NAMES[k]} (median {stats['len_by_class_median'][k]:.0f})")
ax.axvline(data["max_len"], color=MUTED, linestyle="--", linewidth=1)
ax.text(data["max_len"] + 3, ax.get_ylim()[1] * 0.9, f"truncation\n{data['max_len']} tokens\n"
        f"({stats['len_train']['truncated_frac']:.1%} cut)", color=INK2, fontsize=8, va="top")
ax.set_xlabel("Review length (tokens)")
ax.set_ylabel("Density")
ax.set_title("Review length by sentiment class")
ax.legend(fontsize=8, loc="upper left")
save(fig, os.path.join(FIG, "eda_lengths.png"))

# 3) deduplication / leakage
fig, ax = plt.subplots(figsize=(5.2, 2.6))
s = split_stats
rows = [("Official train", s["raw_train"], s["dedup_train"]),
        ("Official test", s["raw_test"], s["test_after_leakage_removal"])]
for y, (name, raw, kept) in enumerate(rows):
    ax.barh(y, raw / 1000, color="#e1e0d9", height=0.55)
    ax.barh(y, kept / 1000, color="#2a78d6", height=0.55)
    ax.text(kept / 1000 + 2, y, f"{kept:,} kept", va="center", color="#ffffff" if False else INK2, fontsize=9)
    ax.text(raw / 1000 + 2, y, f"{raw:,}", va="center", color=MUTED, fontsize=9)
ax.set_yticks([0, 1], [r[0] for r in rows])
ax.invert_yaxis()
ax.set_xlabel("Reviews (thousands)")
ax.set_title("Duplicated reviews removed (gray = dropped)")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, 185)
save(fig, os.path.join(FIG, "eda_dedup.png"))
