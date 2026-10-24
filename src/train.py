"""Train / evaluate one LSTM-variant classifier. Results are written to runs/<group>/<name>/.

Example:
    python src/train.py --cell cifg --hidden 256 --epochs 8 --name cifg_h256 --group main
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from data import LABELS, make_loader, prepare, stratified_subset
from model import LSTMClassifier, count_parameters

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cell", default="vanilla", choices=["vanilla", "wmc", "cifg", "lstmp"])
    p.add_argument("--emb", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--proj", type=int, default=128, help="projection size P (LSTMP only)")
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--init", default="default", choices=["default", "xavier", "orthogonal"])
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "rmsprop", "sgd"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3, help="early stopping on validation macro-F1")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--class-weights", default="sqrt_inv", choices=["none", "sqrt_inv", "inv"])
    p.add_argument("--train-size", type=int, default=None, help="stratified subset of the training set")
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--group", default="debug")
    p.add_argument("--name", default=None)
    p.add_argument("--max-batches", type=int, default=None, help="debug: limit batches per epoch")
    return p.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_optimizer(name, params, lr, wd):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd or 1e-2)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)


def class_weights(labels, mode, n_classes):
    if mode == "none":
        return None
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    w = 1.0 / counts if mode == "inv" else 1.0 / np.sqrt(counts)
    return torch.tensor(w / (w * counts / counts.sum()).sum(), dtype=torch.float)  # mean weight per sample = 1


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    preds, gold, total_loss, n = [], [], 0.0, 0
    for ids, lengths, y in loader:
        ids, y = ids.to(device), y.to(device)
        logits = model(ids, lengths)
        total_loss += loss_fn(logits, y).item() * len(y)
        n += len(y)
        preds.append(logits.argmax(-1).cpu())
        gold.append(y.cpu())
    preds, gold = torch.cat(preds).numpy(), torch.cat(gold).numpy()
    return {"loss": total_loss / n, "acc": accuracy_score(gold, preds),
            "macro_f1": f1_score(gold, preds, average="macro"),
            "f1_per_class": f1_score(gold, preds, average=None).tolist(),
            "confusion": confusion_matrix(gold, preds).tolist()}, preds


def main(argv=None):
    args = parse_args(argv)
    name = args.name or f"{args.cell}_h{args.hidden}_{args.optimizer}_lr{args.lr}_bs{args.batch_size}_{args.init}_s{args.seed}"
    out_dir = os.path.join(ROOT, "runs", args.group, name)
    if os.path.exists(os.path.join(out_dir, "results.json")):
        print(f"[skip] {out_dir} already done")
        return
    os.makedirs(out_dir, exist_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)

    data = prepare(os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "data", "processed", "drugs.pt"),
                   max_len=args.max_len)
    train_idx = stratified_subset(data["train"]["labels"], args.train_size, seed=0)
    train_loader = make_loader(data["train"], args.batch_size, shuffle=True, indices=train_idx, seed=args.seed)
    val_loader = make_loader(data["val"], 256, shuffle=False)
    test_loader = make_loader(data["test"], 256, shuffle=False)

    model = LSTMClassifier(len(data["vocab"]), len(LABELS), cell=args.cell, emb_dim=args.emb,
                           hidden_size=args.hidden, proj_size=args.proj, num_layers=args.layers,
                           bidirectional=args.bidirectional, dropout=args.dropout, init=args.init).to(device)
    n_params = count_parameters(model)
    train_labels = [data["train"]["labels"][i] for i in train_idx]
    w = class_weights(train_labels, args.class_weights, len(LABELS))
    loss_fn = nn.CrossEntropyLoss(weight=None if w is None else w.to(device))
    opt = make_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)

    print(f"{name} | device={device} | train={len(train_idx)} | params={n_params}", flush=True)
    history, best_f1, best_epoch, bad = [], -1.0, 0, 0
    ckpt = os.path.join(out_dir, "best.pt")
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot_loss, tot_correct, n, grad_norms = time.time(), 0.0, 0, 0, []
        for b, (ids, lengths, y) in enumerate(train_loader):
            if args.max_batches and b >= args.max_batches:
                break
            ids, y = ids.to(device), y.to(device)
            logits = model(ids, lengths)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), args.clip if args.clip > 0 else float("inf"))
            grad_norms.append(gn.item())
            opt.step()
            tot_loss += loss.item() * len(y)
            tot_correct += (logits.argmax(-1) == y).sum().item()
            n += len(y)
            if not np.isfinite(loss.item()):
                break
        val, _ = evaluate(model, val_loader, device, loss_fn)
        rec = {"epoch": epoch, "train_loss": tot_loss / n, "train_acc": tot_correct / n,
               "val_loss": val["loss"], "val_acc": val["acc"], "val_macro_f1": val["macro_f1"],
               "grad_norm": float(np.mean(grad_norms)), "epoch_time": time.time() - t0}
        history.append(rec)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = val["macro_f1"], epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                break
        if not np.isfinite(rec["train_loss"]):
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    val, _ = evaluate(model, val_loader, device, loss_fn)
    test, test_preds = evaluate(model, test_loader, device, loss_fn)
    results = {"name": name, "group": args.group, "config": vars(args), "params": n_params,
               "best_epoch": best_epoch, "train_time_s": time.time() - t_start, "history": history,
               "val": val, "test": test, "device": str(device)}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    np.save(os.path.join(out_dir, "test_preds.npy"), test_preds)
    print(f"[done] {name} best_epoch={best_epoch} val_f1={val['macro_f1']:.4f} "
          f"test_f1={test['macro_f1']:.4f} test_acc={test['acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
