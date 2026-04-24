import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from facecheck.data.transforms import AugmentConfig, augment_pair
from facecheck.data.utils import (
    DepthMinMax,
    bgr_to_rgb_224_01,
    compute_depth_minmax,
    iter_subject_dirs,
    load_landmarks_json,
    read_depth_224,
    read_bgr,
    to_1d_tensor,
    to_4ch_tensor,
)


@dataclass(frozen=True)
class Sample:
    rgb_path: str
    depth_path: str
    landmark_key: str
    label: int
    subject: str
    category: str
    subject_dir: str


def _label_from_category(category: str) -> int:
    if category == "affected":
        return 1
    if category == "unaffected":
        return 0
    raise ValueError(category)


class PalsynetDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        seed: int = 42,
        augment: Optional[AugmentConfig] = None,
        depth_minmax: Optional[DepthMinMax] = None,
        landmark_mean: Optional[np.ndarray] = None,
        landmark_std: Optional[np.ndarray] = None,
        landmark_dim: int = 27,
        require_landmarks: bool = True,
        dynaface_fallback: bool = True,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.seed = int(seed)
        self.augment = augment or AugmentConfig(enable=(split == "train"))
        self.landmark_dim = int(landmark_dim)
        self.require_landmarks = bool(require_landmarks)
        self.dynaface_fallback = bool(dynaface_fallback)
        self.landmark_mean = None if landmark_mean is None else np.asarray(landmark_mean, dtype=np.float32).reshape(-1)
        self.landmark_std = None if landmark_std is None else np.asarray(landmark_std, dtype=np.float32).reshape(-1)

        subjects_by_category: Dict[str, List[Tuple[str, str]]] = {"affected": [], "unaffected": []}
        for category, subject, sdir in iter_subject_dirs(dataset_root):
            subjects_by_category[category].append((subject, sdir))

        rng = np.random.default_rng(self.seed)
        test_subjects: Dict[str, set] = {"affected": set(), "unaffected": set()}
        val_subjects: Dict[str, set] = {"affected": set(), "unaffected": set()}
        for category, items in subjects_by_category.items():
            subs = [s for s, _ in items]
            rng.shuffle(subs)
            n = len(subs)
            k_test = int(round(n * self.test_ratio))
            k_val = int(round(n * self.val_ratio))
            k_test = min(max(k_test, 0), n)
            k_val = min(max(k_val, 0), n - k_test)
            if n > 0 and (n - (k_test + k_val)) == 0:
                if k_val > 0:
                    k_val -= 1
                elif k_test > 0:
                    k_test -= 1
            test_subjects[category] = set(subs[:k_test])
            val_subjects[category] = set(subs[k_test : k_test + k_val])

        selected: List[Tuple[str, str, str]] = []
        for category, subject, sdir in iter_subject_dirs(dataset_root):
            is_test = subject in test_subjects[category]
            is_val = subject in val_subjects[category]
            if split == "test" and is_test:
                selected.append((category, subject, sdir))
            if split in {"val", "valid", "validation"} and is_val:
                selected.append((category, subject, sdir))
            if split == "train" and (not is_test) and (not is_val):
                selected.append((category, subject, sdir))

        self.landmarks_by_subject: Dict[str, Dict[str, np.ndarray]] = {}
        samples: List[Sample] = []

        for category, subject, sdir in selected:
            img_dir = os.path.join(sdir, "cropped_img") if os.path.isdir(os.path.join(sdir, "cropped_img")) else sdir
            depth_dir = os.path.join(sdir, "depth") if os.path.isdir(os.path.join(sdir, "depth")) else sdir
            lm_path = os.path.join(sdir, "landmarks.json")

            lm_map: Dict[str, np.ndarray] = {}
            if os.path.isfile(lm_path):
                lm_map = load_landmarks_json(lm_path)
            self.landmarks_by_subject[sdir] = lm_map

            if not os.path.isdir(img_dir) or not os.path.isdir(depth_dir):
                continue
            imgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            imgs.sort()
            for img_name in imgs:
                stem = os.path.splitext(img_name)[0]
                rgb_path = os.path.join(img_dir, img_name)
                depth_path = os.path.join(depth_dir, stem + ".npy")
                if not os.path.isfile(depth_path):
                    continue
                samples.append(
                    Sample(
                        rgb_path=os.path.abspath(rgb_path),
                        depth_path=os.path.abspath(depth_path),
                        landmark_key=img_name,
                        label=_label_from_category(category),
                        subject=subject,
                        category=category,
                        subject_dir=os.path.abspath(sdir),
                    )
                )

        if not samples:
            raise ValueError(f"No samples found under {dataset_root} for split={split}")

        self.samples = samples

        if depth_minmax is None:
            depth_minmax = compute_depth_minmax([s.depth_path for s in self.samples])
        self.depth_minmax = depth_minmax

    def __len__(self) -> int:
        return len(self.samples)

    def _get_landmark_vec(self, sample: Sample) -> Optional[np.ndarray]:
        lm_map = self.landmarks_by_subject.get(sample.subject_dir, {})
        v = lm_map.get(sample.landmark_key)
        if v is None:
            stem = os.path.splitext(sample.landmark_key)[0]
            v = lm_map.get(stem)
        if v is None:
            return None
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.size != self.landmark_dim:
            return None
        return v

    def _dynaface_landmark_vec(self, img_bgr: np.ndarray) -> np.ndarray:
        try:
            import dynaface.facial as facial  # type: ignore
        except Exception:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            lib = os.path.join(root, "dynaface", "dynaface-lib")
            if os.path.isdir(lib) and lib not in sys.path:
                sys.path.insert(0, lib)
            import dynaface.facial as facial  # type: ignore

        lms = facial.infer_landmarks_bgr(img_bgr, crop=False)
        if not lms:
            return np.zeros((self.landmark_dim,), dtype=np.float32)
        h, w = img_bgr.shape[:2]
        arr = np.asarray(lms, dtype=np.float32)
        arr[:, 0] = arr[:, 0] / max(float(w), 1.0)
        arr[:, 1] = arr[:, 1] / max(float(h), 1.0)
        flat = arr.reshape(-1)
        if flat.size < self.landmark_dim:
            pad = np.zeros((self.landmark_dim - flat.size,), dtype=np.float32)
            flat = np.concatenate([flat, pad], axis=0)
        return flat[: self.landmark_dim].astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        img_bgr = read_bgr(s.rgb_path)
        rgb = bgr_to_rgb_224_01(img_bgr)
        depth = read_depth_224(s.depth_path)
        rgb, depth = augment_pair(rgb, depth, self.augment, seed=self.seed + idx)
        depth = self.depth_minmax.normalize(depth)
        x = to_4ch_tensor(rgb, depth)

        lm = self._get_landmark_vec(s)
        if lm is None:
            if self.dynaface_fallback:
                lm = self._dynaface_landmark_vec(img_bgr)
            elif self.require_landmarks:
                raise ValueError(f"Missing/invalid landmarks for {s.rgb_path}")
            else:
                lm = np.zeros((self.landmark_dim,), dtype=np.float32)
        if self.landmark_mean is not None and self.landmark_std is not None:
            denom = np.where(self.landmark_std == 0, 1.0, self.landmark_std)
            lm = (lm - self.landmark_mean) / denom

        return {
            "x": x,
            "landmark": to_1d_tensor(lm),
            "y": torch.tensor(s.label, dtype=torch.long),
        }
