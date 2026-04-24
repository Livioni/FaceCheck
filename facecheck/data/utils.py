import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


def read_bgr(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(path)
    return img_bgr


def bgr_to_rgb_224_01(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    return img_rgb.astype(np.float32) / 255.0


def read_rgb_224(path: str) -> np.ndarray:
    return bgr_to_rgb_224_01(read_bgr(path))


def read_depth_224(path: str) -> np.ndarray:
    depth = np.load(path)
    if depth.ndim == 3:
        depth = depth.squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Invalid depth shape: {depth.shape}")
    depth = cv2.resize(depth.astype(np.float32), (224, 224), interpolation=cv2.INTER_NEAREST)
    return depth


@dataclass(frozen=True)
class DepthMinMax:
    vmin: float
    vmax: float

    def normalize(self, depth: np.ndarray) -> np.ndarray:
        depth = depth.astype(np.float32, copy=False)
        mask = depth > 0
        if not np.any(mask):
            return np.zeros_like(depth, dtype=np.float32)
        denom = float(self.vmax - self.vmin) if float(self.vmax - self.vmin) != 0.0 else 1.0
        out = np.zeros_like(depth, dtype=np.float32)
        out[mask] = (depth[mask] - float(self.vmin)) / denom
        out[mask] = np.clip(out[mask], 0.0, 1.0)
        return out


def compute_depth_minmax(depth_paths: Sequence[str]) -> DepthMinMax:
    vmin = np.inf
    vmax = -np.inf
    for p in depth_paths:
        d = np.load(p)
        if d.ndim == 3:
            d = d.squeeze()
        valid = d[d > 0]
        if valid.size == 0:
            continue
        vmin = min(vmin, float(np.min(valid)))
        vmax = max(vmax, float(np.max(valid)))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    return DepthMinMax(vmin=float(vmin), vmax=float(vmax))


def compute_landmark_mean_std(vectors: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    xs = [np.asarray(v, dtype=np.float32).reshape(-1) for v in vectors]
    if not xs:
        raise ValueError("No landmark vectors to compute stats")
    x = np.stack(xs, axis=0)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def load_landmarks_json(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    out: Dict[str, np.ndarray] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (list, tuple)):
                arr = np.asarray(v, dtype=np.float32)
                if arr.ndim == 1:
                    out[str(k)] = arr
        if out:
            return out

        for key in ("frames", "items", "data"):
            v = obj.get(key)
            if isinstance(v, list):
                obj = v
                break

    if isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict):
                continue
            name = item.get("image") or item.get("img") or item.get("filename") or item.get("frame")
            feat = item.get("landmark") or item.get("landmarks") or item.get("feature") or item.get("features")
            if name is None or feat is None:
                continue
            if isinstance(feat, (list, tuple)):
                arr = np.asarray(feat, dtype=np.float32)
                if arr.ndim == 1:
                    out[str(name)] = arr
    return out


def to_4ch_tensor(rgb_01: np.ndarray, depth_01: np.ndarray) -> torch.Tensor:
    if rgb_01.shape != (224, 224, 3):
        raise ValueError(f"Invalid rgb shape: {rgb_01.shape}")
    if depth_01.shape != (224, 224):
        raise ValueError(f"Invalid depth shape: {depth_01.shape}")
    x = np.concatenate([rgb_01, depth_01[..., None]], axis=-1)
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).to(torch.float32)


def to_1d_tensor(x: np.ndarray) -> torch.Tensor:
    if x.ndim != 1:
        x = x.reshape(-1)
    return torch.from_numpy(x.astype(np.float32, copy=False))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def iter_subject_dirs(dataset_root: str) -> Iterable[Tuple[str, str, str]]:
    for category in ("affected", "unaffected"):
        cdir = os.path.join(dataset_root, category)
        if not os.path.isdir(cdir):
            continue
        for subject in sorted(os.listdir(cdir)):
            sdir = os.path.join(cdir, subject)
            if os.path.isdir(sdir):
                yield category, subject, sdir
