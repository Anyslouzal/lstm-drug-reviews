"""Aggregate all runs into tables (results/*.csv, results/summary.json) and figures (figures/*.png)."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from plotstyle import CELL_COLORS, CELL_NAMES, CLASS_NAMES, GRID as GRID_C, INK, INK2, MUTED, plt, save, setup  # noqa: E402

FIG, RES = os.path.join(ROOT, "figures"), os.path.join(ROOT, "results")
setup()


def load_runs():
    rows = []
    for path in glob.glob(os.path.join(ROOT, "runs", "*", "*", "results.json")):
        r = json.load(open(path))
        c = r["config"]
        rows.append({
            "group": r["group"], "name": r["name"], "cell": c["cell"], "hidden": c["hidden"],
            "proj": c["proj"] if c["cell"] == "lstmp" else None, "seed": c["seed"], "lr": c["lr"],
            "optimizer": c["optimizer"], "init": c["init"], "batch_size": c["batch_size"],
            "layers": c["layers"], "bidirectional": c["bidirectional"],
            "params_recurrent": r["params"]["recurrent"], "params_total": r["params"]["total"],
            "best_epoch": r["best_epoch"], "epochs_run": len(r["history"]),
            "epoch_to_62": next((h["epoch"] for h in r["history"] if h["val_macro_f1"] >= 0.62), np.nan),
            "sec_per_epoch": float(np.mean([h["epoch_time"] for h in r["history"]])),
            "val_f1": r["val"]["macro_f1"], "val_acc": r["val"]["acc"],
            "test_f1": r["test"]["macro_f1"], "test_acc": r["test"]["acc"],
            "test_f1_neg": r["test"]["f1_per_class"][0], "test_f1_neu": r["test"]["f1_per_class"][1],
            "test_f1_pos": r["test"]["f1_per_class"][2],
            "history": r["history"], "confusion": r["test"]["confusion"],
        })
    return pd.DataFrame(rows)


def mean_std(df, by):
    metrics = ["val_f1", "test_f1", "test_acc", "test_f1_neg", "test_f1_neu", "test_f1_pos", "best_epoch",
               "epoch_to_62", "sec_per_epoch"]
    g = df.groupby(by)
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["params_recurrent"] = g["params_recurrent"].first()
    out["n_seeds"] = g.size()
    return out.reset_index()


def fmt(m, s, pct=True):
    k = 100 if pct else 1
    return f"{m * k:.2f} ± {s * k:.2f}" if not np.isnan(s) else f"{m * k:.2f}"


def budget_figure(t, lp, path, figsize, fs):
    """(a) test macro-F1 of each configuration, equal-budget rows first; (b) LSTMP projection size sweep."""
    rows = [("vanilla", 215), ("wmc", 172), ("cifg", 256), ("lstmp", 256), (None, None), ("vanilla", 256), ("wmc", 256)]
    with plt.rc_context({"font.size": fs, "axes.titlesize": fs + 1}):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.35, 1]})
        labels, y = [], 0
        for c, h in rows:
            if c is None:
                y += 0.6
                continue
            r = t[(t.cell == c) & (t.hidden == h)].iloc[0]
            ax.errorbar(r.test_f1_mean * 100, -y, xerr=r.test_f1_std * 100, color=CELL_COLORS[c], marker="o",
                        markersize=fs * 0.6, capsize=4, linewidth=2)
            hp = "H=256, P=128" if c == "lstmp" else f"H={h}"
            labels.append((-y, f"{CELL_NAMES[c].replace('-LSTM', '').replace(' LSTM', '')} ({hp}) · {r.params_recurrent / 1000:.0f}k"))
            y += 1
        ax.set_yticks([p for p, _ in labels], [l for _, l in labels])
        ax.axhline(-3.8, color=GRID_C, linewidth=1)
        ax.text(1.0, -0.02, "equal budget ≈ 296k", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                color=INK2, fontsize=fs - 2)
        ax.text(1.0, -4.62, "larger", transform=ax.get_yaxis_transform(), ha="right", va="bottom", color=INK2,
                fontsize=fs - 2)
        ax.set_xlabel("Test macro-F1 (%), mean ± std over 3 seeds")
        ax.set_title("(a) Same budget, different cells")
        ax.grid(axis="y", visible=False)
        lp = lp.sort_values("proj")
        bx.plot(lp.proj, lp.val_f1 * 100, color="#f5c55b", marker="o", markersize=fs * 0.55, markerfacecolor="#ffffff",
                markeredgewidth=2, label="validation")
        bx.plot(lp.proj, lp.test_f1 * 100, color=CELL_COLORS["lstmp"], marker="o", markersize=fs * 0.55, label="test")
        bx.set_xscale("log", base=2)
        bx.set_xticks(lp.proj, [f"{int(p)}" for p in lp.proj])
        bx.minorticks_off()
        bx.set_xlabel("Projection size P (H = 256)")
        bx.set_ylabel("Macro-F1 (%)")
        bx.set_title("(b) LSTMP projection size (seed 1)")
        bx.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        save(fig, path)


def hparam_figure(panels, path, figsize, fs):
    """Dot plot (validation hollow, test filled) per factor; the base configuration is labelled in bold."""
    with plt.rc_context({"font.size": fs, "axes.titlesize": fs + 1}):
        fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharey=True,
                                 gridspec_kw={"width_ratios": [1, 1.35, 0.8, 1]})
        for ax, (key, title, d) in zip(axes, panels):
            x = np.arange(len(d))
            ordered = key in ("lr", "batch")
            ls = "-" if ordered else ""
            ax.plot(x, d.val_f1 * 100, linestyle=ls, color="#86b6ef", marker="o", markersize=fs * 0.75, markerfacecolor="#ffffff",
                    markeredgewidth=2, label="validation")
            ax.plot(x, d.test_f1 * 100, linestyle=ls, color="#2a78d6", marker="o", markersize=fs * 0.75, label="test")
            ax.set_xticks(x, d.label)
            for lab, b in zip(ax.get_xticklabels(), d.is_base):
                if b:
                    lab.set_fontweight("bold")
                    lab.set_color(INK)
            ax.set_xlim(-0.5, len(d) - 0.5)
            ax.set_title(title)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("Macro-F1 (%)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2, frameon=False,
                   bbox_to_anchor=(1.0, 1.06))
        fig.text(0.0, -0.04, "Bold = base configuration (Adam, lr 1e-3, batch 128, default init); one factor changed at a time.",
                 color=INK2, fontsize=fs - 1)
        fig.tight_layout()
        save(fig, path)


def main():
    df = load_runs()
    if df.empty:
        print("no runs yet")
        return
    print(df.groupby("group").size())
    summary = {}
    order = [c for c in ["vanilla", "wmc", "cifg", "lstmp"] if c in set(df.cell)]

    # ---------- main comparison (H=256)
    main_df = df[df.group == "main"]
    if len(main_df):
        t = mean_std(main_df, "cell").set_index("cell").loc[[c for c in order if c in set(main_df.cell)]]
        t.to_csv(os.path.join(RES, "main_comparison.csv"))
        summary["main"] = t.reset_index().to_dict(orient="records")
        print("\nMAIN (H=256)\n" + "\n".join(
            f"{c:8s} params={int(r.params_recurrent):>7,}  val F1 {fmt(r.val_f1_mean, r.val_f1_std)}  "
            f"test F1 {fmt(r.test_f1_mean, r.test_f1_std)}  acc {fmt(r.test_acc_mean, r.test_acc_std)}  "
            f"neu F1 {fmt(r.test_f1_neu_mean, r.test_f1_neu_std)}  ep {r.best_epoch_mean:.1f}  to62 {r.epoch_to_62_mean:.1f}  "
            f"{r.sec_per_epoch_mean:.0f}s/ep  n={r.n_seeds}" for c, r in t.iterrows()))

        # learning curves (seed 1)
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
        for c in order:
            runs = main_df[main_df.cell == c]
            if not len(runs):
                continue
            H = [pd.DataFrame(h) for h in runs.history]
            n = min(len(h) for h in H)
            vl = np.mean([h.val_loss.values[:n] for h in H], 0)
            tl = np.mean([h.train_loss.values[:n] for h in H], 0)
            vf = np.mean([h.val_macro_f1.values[:n] for h in H], 0)
            ep = np.arange(1, n + 1)
            axes[0].plot(ep, tl, color=CELL_COLORS[c], linestyle=":", linewidth=1.5)
            axes[0].plot(ep, vl, color=CELL_COLORS[c], label=CELL_NAMES[c])
            axes[1].plot(ep, vf * 100, color=CELL_COLORS[c], marker="o", markersize=4, label=CELL_NAMES[c])
        axes[0].set_title("Loss (solid: validation, dotted: train)")
        axes[0].set_xlabel("Epoch")
        axes[1].set_title("Validation macro-F1 (%)")
        axes[1].set_xlabel("Epoch")
        axes[1].legend(fontsize=8, loc="lower right")
        save(fig, os.path.join(FIG, "learning_curves.png"))

    # ---------- parameter / performance trade-off (main + matched)
    pm = df[df.group.isin(["main", "matched"])]
    if len(pm):
        t = mean_std(pm, ["cell", "hidden"])
        t.to_csv(os.path.join(RES, "params_vs_f1.csv"), index=False)
        summary["params_vs_f1"] = t.to_dict(orient="records")
        print("\nPARAMS vs F1\n" + t[["cell", "hidden", "params_recurrent", "test_f1_mean", "test_f1_std",
                                      "n_seeds"]].to_string(index=False))

    # ---------- LSTMP projection size
    lp = df[(df.cell == "lstmp") & (df.group.isin(["lstmp_proj", "main"])) & (df.seed == 1)]
    if len(lp) > 1:
        lp = lp.sort_values("proj")
        summary["lstmp_proj"] = lp[["proj", "params_recurrent", "val_f1", "test_f1", "test_acc"]].to_dict(orient="records")
        print("\nLSTMP proj\n" + lp[["proj", "params_recurrent", "val_f1", "test_f1"]].to_string(index=False))
    if len(pm) and len(lp) > 1:
        budget_figure(t, lp, os.path.join(FIG, "params_vs_f1.png"), (9.2, 3.0), 9)
        budget_figure(t, lp, os.path.join(FIG, "slide_budget.png"), (15, 5.4), 17)

    # ---------- hyper-parameter study on vanilla (base config = main/vanilla_s1)
    hp = df[df.group == "hparams"]
    base = df[(df.group == "main") & (df.cell == "vanilla") & (df.seed == 1)]
    if len(hp):
        hp = pd.concat([hp, base])
        studies = [("lr", "Learning rate (Adam)", lambda d: d[(d.optimizer == "adam") & (d.batch_size == 128) & (d.init == "default")], "lr"),
                   ("optimizer", "Optimizer", lambda d: d[(d.batch_size == 128) & (d.init == "default") & (((d.optimizer == "adam") & (d.lr == 1e-3)) | (d.optimizer != "adam"))], "optimizer"),
                   ("init", "Weight initialisation", lambda d: d[(d.optimizer == "adam") & (d.lr == 1e-3) & (d.batch_size == 128)], "init"),
                   ("batch", "Batch size", lambda d: d[(d.optimizer == "adam") & (d.lr == 1e-3) & (d.init == "default")], "batch_size")]
        summary["hparams"] = {}
        panels = []
        for key, title, sel, col in studies:
            d = sel(hp).copy()
            if key == "optimizer":
                short = {"adam": "Adam", "adamw": "AdamW", "rmsprop": "RMS-\nprop"}
                d["label"] = d.apply(lambda r: short.get(r.optimizer, f"SGD\n{r.lr:g}"), axis=1)
                d = d.sort_values(["optimizer", "lr"])
            else:
                d = d.sort_values(col)
                names = {"default": "default", "orthogonal": "ortho-\ngonal", "xavier": "Xavier"}
                d["label"] = d[col].map(lambda v: f"{v:.0e}".replace("e-0", "e-") if isinstance(v, float)
                                        else names.get(v, str(v)))
            d["is_base"] = (d.optimizer == "adam") & (d.lr == 1e-3) & (d.batch_size == 128) & (d.init == "default")
            summary["hparams"][key] = d[["label", "val_f1", "test_f1", "test_acc", "best_epoch", "sec_per_epoch"]].to_dict(orient="records")
            print(f"\nHP {key}\n" + d[["label", "val_f1", "test_f1", "best_epoch", "sec_per_epoch"]].to_string(index=False))
            panels.append((key, title, d))
        hparam_figure(panels, os.path.join(FIG, "hparams.png"), (11, 2.8), 9)
        hparam_figure(panels, os.path.join(FIG, "slide_hparams.png"), (15, 5.0), 18)

    # ---------- learning-rate sensitivity per cell
    lc = df[(df.group == "lr_cells") | ((df.group == "main") & (df.seed == 1)) |
            ((df.group == "hparams") & df.name.str.startswith("lr_"))]
    lc = lc[(lc.optimizer == "adam") & (lc.batch_size == 128) & (lc.init == "default") & (lc.hidden == 256) &
            (lc.proj.isna() | (lc.proj == 128))]
    if (df.group == "lr_cells").any():
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        summary["lr_cells"] = {}
        for c in order:
            r = lc[(lc.cell == c) & lc.lr.isin([3e-4, 1e-3, 3e-3])].sort_values("lr")
            summary["lr_cells"][c] = r[["lr", "val_f1", "test_f1"]].to_dict(orient="records")
            if len(r) < 2:
                continue
            ax.plot(r.lr, r.val_f1 * 100, color=CELL_COLORS[c], marker="o", markersize=6, label=CELL_NAMES[c])
        ax.set_xscale("log")
        ax.set_xticks([3e-4, 1e-3, 3e-3], ["3e-4", "1e-3", "3e-3"])
        ax.set_xlabel("Learning rate (Adam)")
        ax.set_ylabel("Validation macro-F1 (%)")
        ax.set_title("Learning-rate sensitivity of each cell")
        ax.legend(fontsize=8)
        save(fig, os.path.join(FIG, "lr_cells.png"))
        print("\nLR per cell\n" + lc[["cell", "lr", "val_f1", "test_f1"]].sort_values(["cell", "lr"]).to_string(index=False))

    # ---------- final model
    fin = df[df.group == "final"]
    if len(fin):
        summary["final"] = fin.drop(columns=["history"]).to_dict(orient="records")
        print("\nFINAL\n" + fin[["name", "params_recurrent", "val_f1", "test_f1", "test_acc", "test_f1_neg",
                                  "test_f1_neu", "test_f1_pos"]].to_string(index=False))
        best = fin.sort_values("val_f1").iloc[-1]
        cm = np.array(best.confusion, dtype=float)
        cmn = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(3.6, 3.1))
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmn[i, j]:.0%}\n({int(cm[i, j]):,})", ha="center", va="center", fontsize=8,
                        color="#ffffff" if cmn[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), CLASS_NAMES)
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)
        ax.set_title("Final model: test confusion (row-normalised)", fontsize=9)
        save(fig, os.path.join(FIG, "confusion_final.png"))

    df.drop(columns=["history", "confusion"]).to_csv(os.path.join(RES, "all_runs.csv"), index=False)
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)


if __name__ == "__main__":
    main()
