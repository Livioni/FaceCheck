import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from facecheck.data.dataset import PalsynetDataset
from facecheck.data.utils import DepthMinMax
from facecheck.models.vit_facecheck import FaceCheckViTConfig, ViTFaceCheck


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
def evaluate(model: ViTFaceCheck, loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float]]:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--use_dynaface_landmarks", action="store_true")
    args = ap.parse_args()

    has_landmarks_json = False
    for root, _, files in os.walk(args.dataset_root):
        if "landmarks.json" in files:
            has_landmarks_json = True
            break
    use_dynaface = (not has_landmarks_json) and bool(args.use_dynaface_landmarks)
    if use_dynaface and args.num_workers != 0:
        args.num_workers = 0

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    depth_minmax = ckpt.get("depth_minmax", {"vmin": 0.0, "vmax": 1.0})
    lm_mean = ckpt.get("landmark_mean")
    lm_std = ckpt.get("landmark_std")

    model_cfg = FaceCheckViTConfig(
        backbone=str(cfg_dict.get("backbone", "vit_base_patch16_224")),
        in_chans=4,
        landmark_dim=int(cfg_dict.get("landmark_dim", 27)),
        landmark_hidden=int(cfg_dict.get("landmark_hidden", 256)),
        dropout=float(cfg_dict.get("dropout", 0.0)),
    )
    model = ViTFaceCheck(model_cfg, pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)

    ds = PalsynetDataset(
        dataset_root=args.dataset_root,
        split="test",
        val_ratio=float(cfg_dict.get("val_ratio", 0.1)),
        test_ratio=float(cfg_dict.get("test_ratio", 0.2)),
        seed=int(cfg_dict.get("seed", 42)),
        augment=None,
        depth_minmax=DepthMinMax(vmin=float(depth_minmax["vmin"]), vmax=float(depth_minmax["vmax"])),
        landmark_mean=None if lm_mean is None else np.asarray(lm_mean, dtype=np.float32),
        landmark_std=None if lm_std is None else np.asarray(lm_std, dtype=np.float32),
        landmark_dim=int(cfg_dict.get("landmark_dim", 27)),
        require_landmarks=False,
        dynaface_fallback=use_dynaface,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss, m = evaluate(model, loader, device)
    print({"loss": loss, **m})


if __name__ == "__main__":
    main()
