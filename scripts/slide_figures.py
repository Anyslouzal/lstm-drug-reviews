"""Single-panel figures with larger type, sized for the slides (figures/slide_*.png)."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from analyze import load_runs  # noqa: E402
from plotstyle import CELL_COLORS, CELL_NAMES, INK2, plt, save, setup  # noqa: E402

FIG = os.path.join(ROOT, "figures")
setup()
plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "legend.fontsize": 13})
ORDER = ["vanilla", "wmc", "cifg", "lstmp"]


def val_curves(df):
    main = df[df.group == "main"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ends = []
    for c in ORDER:
        H = [pd.DataFrame(h) for h in main[main.cell == c].history]
        n = min(len(h) for h in H)
        f = np.array([h.val_macro_f1.values[:n] for h in H]) * 100
        ep = np.arange(1, n + 1)
        ax.fill_between(ep, f.mean(0) - f.std(0), f.mean(0) + f.std(0), color=CELL_COLORS[c], alpha=0.12, linewidth=0)
        ax.plot(ep, f.mean(0), color=CELL_COLORS[c], marker="o", markersize=5, label=CELL_NAMES[c])
        ends.append((f.mean(0)[-1], c))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1 (%)")
    ax.set_title("Validation macro-F1 per epoch (mean ± std, 3 seeds)")
    ax.set_xticks(range(1, 11))
    ax.legend(loc="lower right", frameon=False)
    save(fig, os.path.join(FIG, "slide_val_curves.png"))


def confusion(df):
    fin = df[df.group == "final"].sort_values("val_f1").iloc[-1]
    cm = np.array(fin.confusion, dtype=float)
    cmn = cm / cm.sum(1, keepdims=True)
    names = ["negative", "neutral", "positive"]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]:.0%}\n({int(cm[i, j]):,})", ha="center", va="center", fontsize=14,
                    color="#fcfcfb" if cmn[i, j] > 0.55 else "#0b0b0b")
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    ax.set_title("Test confusion matrix (row-normalised)")
    save(fig, os.path.join(FIG, "slide_confusion.png"))


if __name__ == "__main__":
    df = load_runs()
    val_curves(df)
    confusion(df)
