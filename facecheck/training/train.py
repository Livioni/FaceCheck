import argparse
import csv
import json
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from facecheck.data.dataset import PalsynetDataset
from facecheck.data.utils import DepthMinMax, compute_depth_minmax, compute_landmark_mean_std, ensure_dir
from facecheck.models.vit_facecheck import FaceCheckViTConfig, ViTFaceCheck
from facecheck.training.config import TrainConfig


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    out: Dict[str, float] = {}
    out["acc"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auc"] = float("nan")
    return out


@torch.no_grad()
def _eval(model: ViTFaceCheck, loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    ys = []
    ps = []
    losses = []
    ce = torch.nn.CrossEntropyLoss()
    for batch in loader:
        x = batch["x"].to(device)
        lm = batch["landmark"].to(device)
        y = batch["y"].to(device)
        logits = model(x, lm)
        loss = ce(logits, y)
        prob = torch.softmax(logits, dim=1)[:, 1]
        ys.append(y.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())
        losses.append(float(loss.detach().cpu().item()))
    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)
    return float(np.mean(losses) if losses else 0.0), _metrics(y_true, y_prob)


def _compute_stats(ds_train_raw: PalsynetDataset) -> Tuple[DepthMinMax, np.ndarray, np.ndarray]:
    depth_mm = compute_depth_minmax([s.depth_path for s in ds_train_raw.samples])
    lms = []
    for s in ds_train_raw.samples:
        v = ds_train_raw._get_landmark_vec(s)
        if v is not None:
            lms.append(v)
    if not lms:
        lm_mean = np.zeros((ds_train_raw.landmark_dim,), dtype=np.float32)
        lm_std = np.ones((ds_train_raw.landmark_dim,), dtype=np.float32)
    else:
        lm_mean, lm_std = compute_landmark_mean_std(lms)
    return depth_mm, lm_mean, lm_std


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--output_dir", default="outputs/facecheck")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--use_dynaface_landmarks", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--backbone", default="vit_base_patch16_224")
    ap.add_argument("--pretrained_ckpt", default=None)
    ap.add_argument("--landmark_dim", type=int, default=27)
    ap.add_argument("--landmark_hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args()

    has_landmarks_json = False
    for root, _, files in os.walk(args.dataset_root):
        if "landmarks.json" in files:
            has_landmarks_json = True
            break
    use_dynaface = (not has_landmarks_json) and bool(args.use_dynaface_landmarks)
    if use_dynaface and args.num_workers != 0:
        args.num_workers = 0

    cfg = TrainConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        backbone=args.backbone,
        pretrained_ckpt=args.pretrained_ckpt,
        landmark_dim=args.landmark_dim,
        landmark_hidden=args.landmark_hidden,
        dropout=args.dropout,
    )

    ensure_dir(cfg.output_dir)
    _set_seed(cfg.seed)
    log_path = os.path.join(cfg.output_dir, "train_log.jsonl")
    cfg_path = os.path.join(cfg.output_dir, "config.json")
    csv_path = os.path.join(cfg.output_dir, "metrics.csv")
    plot_path = os.path.join(cfg.output_dir, "curves.png")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)

    ds_train_raw = PalsynetDataset(
        dataset_root=cfg.dataset_root,
        split="train",
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
        augment=None,
        depth_minmax=None,
        landmark_mean=None,
        landmark_std=None,
        landmark_dim=cfg.landmark_dim,
        require_landmarks=False,
        dynaface_fallback=use_dynaface,
    )
    depth_mm, lm_mean, lm_std = _compute_stats(ds_train_raw)

    ds_train = PalsynetDataset(
        dataset_root=cfg.dataset_root,
        split="train",
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
        augment=None,
        depth_minmax=depth_mm,
        landmark_mean=lm_mean,
        landmark_std=lm_std,
        landmark_dim=cfg.landmark_dim,
        require_landmarks=False,
        dynaface_fallback=use_dynaface,
    )
    ds_val = PalsynetDataset(
        dataset_root=cfg.dataset_root,
        split="val",
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
        augment=None,
        depth_minmax=depth_mm,
        landmark_mean=lm_mean,
        landmark_std=lm_std,
        landmark_dim=cfg.landmark_dim,
        require_landmarks=False,
        dynaface_fallback=use_dynaface,
    )
    ds_test = PalsynetDataset(
        dataset_root=cfg.dataset_root,
        split="test",
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
        augment=None,
        depth_minmax=depth_mm,
        landmark_mean=lm_mean,
        landmark_std=lm_std,
        landmark_dim=cfg.landmark_dim,
        require_landmarks=False,
        dynaface_fallback=use_dynaface,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = FaceCheckViTConfig(
        backbone=cfg.backbone,
        in_chans=4,
        landmark_dim=cfg.landmark_dim,
        landmark_hidden=cfg.landmark_hidden,
        dropout=cfg.dropout,
    )
    model = ViTFaceCheck(model_cfg, pretrained=False)
    if cfg.pretrained_ckpt:
        model.load_pretrained(cfg.pretrained_ckpt, strict=False)
    elif "_dinov3" in cfg.backbone:
        model.load_timm_pretrained()
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = torch.nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0
    history = []

    log_f = open(log_path, "w", encoding="utf-8")
    try:
        init_log = {
            "event": "init",
            "ts": time.time(),
            "output_dir": cfg.output_dir,
            "train_size": len(ds_train),
            "val_size": len(ds_val),
            "test_size": len(ds_test),
        }
        print(json.dumps(init_log, ensure_ascii=False))
        log_f.write(json.dumps(init_log, ensure_ascii=False) + "\n")
        log_f.flush()

        for epoch in range(cfg.epochs):
            model.train()
            losses = []
            t0 = time.time()
            for batch in train_loader:
                x = batch["x"].to(device, non_blocking=True)
                lm = batch["landmark"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                logits = model(x, lm)
                loss = ce(logits, y)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu().item()))

            val_loss, val_m = _eval(model, val_loader, device)
            train_loss = float(np.mean(losses) if losses else 0.0)

            if val_m["f1"] > best_f1:
                best_f1 = val_m["f1"]
                best_epoch = epoch
                bad_epochs = 0
                ckpt = {
                    "model": model.state_dict(),
                    "config": cfg.to_dict(),
                    "depth_minmax": {"vmin": depth_mm.vmin, "vmax": depth_mm.vmax},
                    "landmark_mean": lm_mean.tolist(),
                    "landmark_std": lm_std.tolist(),
                }
                torch.save(ckpt, os.path.join(cfg.output_dir, "best.pt"))
            else:
                bad_epochs += 1

            log = {
                "event": "epoch_end",
                "ts": time.time(),
                "epoch": epoch,
                "sec": float(time.time() - t0),
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"val_{k}": float(v) for k, v in val_m.items()},
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
            }
            history.append(log)
            print(json.dumps(log, ensure_ascii=False))
            log_f.write(json.dumps(log, ensure_ascii=False) + "\n")
            log_f.flush()

            if bad_epochs >= cfg.patience:
                stop_log = {
                    "event": "early_stop",
                    "ts": time.time(),
                    "epoch": epoch,
                    "patience": cfg.patience,
                    "best_epoch": best_epoch,
                    "best_f1": best_f1,
                }
                print(json.dumps(stop_log, ensure_ascii=False))
                log_f.write(json.dumps(stop_log, ensure_ascii=False) + "\n")
                log_f.flush()
                break
    finally:
        log_f.close()

    best_path = os.path.join(cfg.output_dir, "best.pt")
    if os.path.isfile(best_path):
        best = torch.load(best_path, map_location="cpu")
        model.load_state_dict(best["model"], strict=True)
        model.to(device)
        test_loss, test_m = _eval(model, test_loader, device)
        out = {"event": "test_eval", "ts": time.time(), "test_loss": test_loss, **{f"test_{k}": float(v) for k, v in test_m.items()}}
        print(json.dumps(out, ensure_ascii=False))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    if history:
        fieldnames = sorted({k for row in history for k in row.keys()})
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in history:
                w.writerow(row)

        try:
            import matplotlib.pyplot as plt  # type: ignore

            epochs = [int(r["epoch"]) for r in history]
            train_loss = [float(r.get("train_loss", float("nan"))) for r in history]
            val_loss = [float(r.get("val_loss", float("nan"))) for r in history]
            val_f1 = [float(r.get("val_f1", float("nan"))) for r in history]
            val_acc = [float(r.get("val_acc", float("nan"))) for r in history]
            val_precision = [float(r.get("val_precision", float("nan"))) for r in history]
            val_recall = [float(r.get("val_recall", float("nan"))) for r in history]
            val_auc = [float(r.get("val_auc", float("nan"))) for r in history]

            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax0, ax1 = axes

            ax0.plot(epochs, train_loss, label="train_loss")
            ax0.plot(epochs, val_loss, label="val_loss")
            ax0.set_ylabel("loss")
            ax0.grid(True, alpha=0.3)
            ax0.legend()

            ax1.plot(epochs, val_f1, label="val_f1")
            ax1.plot(epochs, val_acc, label="val_acc")
            ax1.plot(epochs, val_precision, label="val_precision")
            ax1.plot(epochs, val_recall, label="val_recall")
            ax1.plot(epochs, val_auc, label="val_auc")
            ax1.set_xlabel("epoch")
            ax1.set_ylabel("metric")
            ax1.grid(True, alpha=0.3)
            ax1.legend(ncol=2)

            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
        except Exception as e:
            msg = {"event": "plot_skip", "ts": time.time(), "error": str(e)}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
