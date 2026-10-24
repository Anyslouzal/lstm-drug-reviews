"""Run every experiment of the study (finished runs are skipped, so the script can be re-launched).

    python scripts/run_experiments.py --groups main hparams --workers 2
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [1, 2, 3]
CELLS = ["vanilla", "wmc", "cifg", "lstmp"]


def experiments():
    exps = {}
    # 1) Architecture comparison: same width H=256 (LSTMP: H=256, P=128), 3 seeds each
    exps["main"] = [(f"{c}_s{s}", ["--cell", c, "--seed", str(s)]) for s in SEEDS for c in CELLS]
    # 2) Same comparison at an equal recurrent-parameter budget (~296k, the size of CIFG-256 and LSTMP-256/128)
    exps["matched"] = [(f"{c}_h{h}_s{s}", ["--cell", c, "--hidden", str(h), "--seed", str(s)])
                       for s in SEEDS for c, h in [("vanilla", 215), ("wmc", 172)]]
    # 3) LSTMP: effect of the projection size P (H fixed at 256)
    exps["lstmp_proj"] = [(f"lstmp_p{p}", ["--cell", "lstmp", "--proj", str(p)]) for p in [32, 64, 192, 256]]
    # 4) Hyper-parameter study on the vanilla baseline (one factor at a time, base = Adam 1e-3, bs 128, default init;
    #    the base configuration itself is main/vanilla_s1)
    hp = []
    hp += [(f"lr_{lr:g}", ["--lr", str(lr)]) for lr in [1e-4, 3e-4, 3e-3, 1e-2]]
    hp += [(f"opt_{o}", ["--optimizer", o]) for o in ["adamw", "rmsprop"]]
    hp += [(f"opt_sgd_lr{lr:g}", ["--optimizer", "sgd", "--lr", str(lr)]) for lr in [0.1, 1.0]]
    hp += [(f"init_{i}", ["--init", i]) for i in ["xavier", "orthogonal"]]
    hp += [(f"bs_{b}", ["--batch-size", str(b)]) for b in [32, 64, 256, 512]]
    exps["hparams"] = hp
    # 5) Learning-rate sensitivity of each non-vanilla cell (to check the comparison is not an artefact of lr=1e-3)
    exps["lr_cells"] = [(f"{c}_lr{lr:g}", ["--cell", c, "--lr", str(lr)])
                        for c in ["wmc", "cifg"] for lr in [3e-4, 3e-3]]
    # 6) Final model: best cell (CIFG) + best settings of the study, read in both directions;
    #    the candidate is chosen on validation macro-F1 only
    final = ["--cell", "cifg", "--bidirectional", "--init", "xavier", "--lr", "2e-3", "--epochs", "12",
             "--patience", "4"]
    exps["final"] = [("cifg_bi_l1", final), ("cifg_bi_l2", final + ["--layers", "2"])]
    return exps


def run(group, name, extra, device):
    out_dir = os.path.join(ROOT, "runs", group, name)
    if os.path.exists(os.path.join(out_dir, "results.json")):
        return f"[skip] {group}/{name}"
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, os.path.join(ROOT, "src", "train.py"), "--group", group, "--name", name,
           "--epochs", "10", "--device", device] + extra
    t = time.time()
    with open(os.path.join(out_dir, "log.txt"), "w") as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    return f"[{'ok' if code == 0 else 'FAIL'}] {group}/{name} ({(time.time() - t) / 60:.1f} min)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", nargs="+", default=["main", "hparams", "matched", "lstmp_proj", "lr_cells"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    exps = experiments()
    jobs = [(g, n, e) for g in args.groups for n, e in exps[g]]
    print(f"{len(jobs)} runs", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for msg in pool.map(lambda j: run(*j, args.device), jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
