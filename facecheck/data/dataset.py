import os
import sys
import csv
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


def _safe_float(x: object) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def _load_measures_csv(path: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return [], {}
        if "image_path" not in reader.fieldnames:
            return [], {}
        keys = [k for k in reader.fieldnames if k != "image_path"]
        out: Dict[str, np.ndarray] = {}
        for row in reader:
            raw_path = row.get("image_path", "")
            base = os.path.basename(str(raw_path))
            if not base:
                continue
            feats = np.asarray([_safe_float(row.get(k)) for k in keys], dtype=np.float32)
            out[base] = feats
            out[os.path.splitext(base)[0]] = feats
        return keys, out


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
        self.measures_by_subject: Dict[str, Dict[str, np.ndarray]] = {}
        self.measure_keys: Optional[List[str]] = None
        samples: List[Sample] = []

        for category, subject, sdir in selected:
            img_dir = os.path.join(sdir, "cropped_img") if os.path.isdir(os.path.join(sdir, "cropped_img")) else sdir
            depth_dir = os.path.join(sdir, "depth") if os.path.isdir(os.path.join(sdir, "depth")) else sdir
            lm_path = os.path.join(sdir, "landmarks.json")
            measures_path = os.path.join(sdir, f"{subject}_measures.csv")
            if not os.path.isfile(measures_path):
                cand = [p for p in os.listdir(sdir) if p.endswith("_measures.csv")]
                if len(cand) == 1:
                    measures_path = os.path.join(sdir, cand[0])
                else:
                    measures_path = ""

            lm_map: Dict[str, np.ndarray] = {}
            if os.path.isfile(lm_path):
                lm_map = load_landmarks_json(lm_path)
            self.landmarks_by_subject[sdir] = lm_map

            meas_map: Dict[str, np.ndarray] = {}
            if measures_path and os.path.isfile(measures_path):
                keys, meas_map = _load_measures_csv(measures_path)
                if self.measure_keys is None and keys:
                    self.measure_keys = keys
            self.measures_by_subject[sdir] = meas_map

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
        meas_map = self.measures_by_subject.get(sample.subject_dir, {})
        v = meas_map.get(sample.landmark_key)
        if v is None:
            stem = os.path.splitext(sample.landmark_key)[0]
            v = meas_map.get(stem)
        if v is not None:
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            if v.size < self.landmark_dim:
                pad = np.zeros((self.landmark_dim - v.size,), dtype=np.float32)
                v = np.concatenate([v, pad], axis=0)
            return v[: self.landmark_dim].astype(np.float32, copy=False)

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
        from facecheck.inference.predictor import _import_dynaface, _ensure_dynaface_models

        try:
            facial, measures, models = _import_dynaface()
            _ensure_dynaface_models(models)
            analyzer = facial.AnalyzeFace(measures=[measures.AnalyzeLandmarks()])
            ok = analyzer.load_image(img_bgr, crop=False)
            if not ok or analyzer.is_no_face():
                return np.zeros((self.landmark_dim,), dtype=np.float32)
            lm_dict = analyzer.analyze()
            if lm_dict is None:
                return np.zeros((self.landmark_dim,), dtype=np.float32)
            n = int(getattr(measures.AnalyzeLandmarks, "NUM_LANDMARKS", 97))
            arr = np.zeros((n, 2), dtype=np.float32)
            for i in range(1, n + 1):
                arr[i - 1, 0] = float(lm_dict.get(f"landmark-{i}-x", 0.0))
                arr[i - 1, 1] = float(lm_dict.get(f"landmark-{i}-y", 0.0))
        except Exception:
            return np.zeros((self.landmark_dim,), dtype=np.float32)

        h, w = img_bgr.shape[:2]
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
        rgb, depth = augment_pair(rgb, depth, self.augment, seed=None)
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


@dataclass(frozen=True)
class HumanFaceSample:
    rgb_path: str
    depth_path: str
    feature_key: str
    label: int


def _pad_or_trunc_1d(arr: Optional[np.ndarray], dim: int, fill: float) -> Optional[np.ndarray]:
    if arr is None:
        return None
    v = np.asarray(arr, dtype=np.float32).reshape(-1)
    if v.size < dim:
        pad = np.full((dim - v.size,), float(fill), dtype=np.float32)
        v = np.concatenate([v, pad], axis=0)
    return v[:dim].astype(np.float32, copy=False)


class HumanFacesDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str,
        csv_path: Optional[str] = None,
        label: int = 0,
        max_samples: Optional[int] = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        seed: int = 42,
        augment: Optional[AugmentConfig] = None,
        depth_minmax: Optional[DepthMinMax] = None,
        landmark_mean: Optional[np.ndarray] = None,
        landmark_std: Optional[np.ndarray] = None,
        landmark_dim: int = 27,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.csv_path = csv_path or os.path.join(dataset_root, "csv", "human_faces_measures.csv")
        self.label = int(label)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.seed = int(seed)
        self.augment = augment or AugmentConfig(enable=(split == "train"))
        self.landmark_dim = int(landmark_dim)
        self.landmark_mean = _pad_or_trunc_1d(landmark_mean, self.landmark_dim, fill=0.0)
        self.landmark_std = _pad_or_trunc_1d(landmark_std, self.landmark_dim, fill=1.0)

        cropped_dir = os.path.join(dataset_root, "cropped_images")
        depth_dir = os.path.join(dataset_root, "depth_results")
        if not os.path.isdir(cropped_dir):
            raise ValueError(f"Missing cropped_images under {dataset_root}")
        if not os.path.isdir(depth_dir):
            raise ValueError(f"Missing depth_results under {dataset_root}")
        if not os.path.isfile(self.csv_path):
            raise ValueError(f"Missing csv file: {self.csv_path}")

        with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"Empty csv header: {self.csv_path}")
            if "image_path" not in reader.fieldnames:
                raise ValueError(f"csv must contain 'image_path' column: {self.csv_path}")
            feature_keys = [k for k in reader.fieldnames if k != "image_path"]
            self.feature_keys = feature_keys

            all_items: List[Tuple[str, str, str, np.ndarray]] = []
            for row in reader:
                raw_path = row.get("image_path", "")
                base = os.path.basename(str(raw_path))
                if not base:
                    continue
                rgb_path = os.path.join(cropped_dir, base)
                if not os.path.isfile(rgb_path):
                    continue
                stem = os.path.splitext(base)[0]
                depth_path = os.path.join(depth_dir, stem + ".npy")
                if not os.path.isfile(depth_path):
                    continue

                feats = np.asarray([_safe_float(row.get(k)) for k in feature_keys], dtype=np.float32)
                all_items.append((base, rgb_path, depth_path, feats))

        if not all_items:
            raise ValueError(f"No valid samples found under {dataset_root} with csv={self.csv_path}")

        rng = np.random.default_rng(self.seed)
        idxs = np.arange(len(all_items))
        rng.shuffle(idxs)
        n = len(idxs)
        k_test = int(round(n * self.test_ratio))
        k_val = int(round(n * self.val_ratio))
        k_test = min(max(k_test, 0), n)
        k_val = min(max(k_val, 0), n - k_test)
        if n > 0 and (n - (k_test + k_val)) == 0:
            if k_val > 0:
                k_val -= 1
            elif k_test > 0:
                k_test -= 1

        test_set = set(idxs[:k_test].tolist())
        val_set = set(idxs[k_test : k_test + k_val].tolist())

        samples: List[HumanFaceSample] = []
        features_by_key: Dict[str, np.ndarray] = {}
        for i, (key, rgb_path, depth_path, feats) in enumerate(all_items):
            is_test = i in test_set
            is_val = i in val_set
            if split == "test" and not is_test:
                continue
            if split in {"val", "valid", "validation"} and not is_val:
                continue
            if split == "train" and (is_test or is_val):
                continue

            samples.append(
                HumanFaceSample(
                    rgb_path=os.path.abspath(rgb_path),
                    depth_path=os.path.abspath(depth_path),
                    feature_key=key,
                    label=self.label,
                )
            )
            features_by_key[key] = feats

        if not samples:
            raise ValueError(f"No samples found under {dataset_root} for split={split}")

        if self.max_samples is not None and self.max_samples > 0 and len(samples) > self.max_samples:
            samples = samples[: self.max_samples]
            keep = {s.feature_key for s in samples}
            features_by_key = {k: v for k, v in features_by_key.items() if k in keep}

        self.samples = samples
        self.features_by_key = features_by_key

        if depth_minmax is None:
            depth_minmax = compute_depth_minmax([s.depth_path for s in self.samples])
        self.depth_minmax = depth_minmax

    def __len__(self) -> int:
        return len(self.samples)

    def _get_feature_vec(self, sample: HumanFaceSample) -> np.ndarray:
        v = self.features_by_key.get(sample.feature_key)
        if v is None:
            v = np.zeros((len(self.feature_keys),), dtype=np.float32)
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.size < self.landmark_dim:
            pad = np.zeros((self.landmark_dim - v.size,), dtype=np.float32)
            v = np.concatenate([v, pad], axis=0)
        return v[: self.landmark_dim].astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        img_bgr = read_bgr(s.rgb_path)
        rgb = bgr_to_rgb_224_01(img_bgr)
        depth = read_depth_224(s.depth_path)
        rgb, depth = augment_pair(rgb, depth, self.augment, seed=None)
        depth = self.depth_minmax.normalize(depth)
        x = to_4ch_tensor(rgb, depth)

        feat = self._get_feature_vec(s)
        if self.landmark_mean is not None and self.landmark_std is not None:
            denom = np.where(self.landmark_std == 0, 1.0, self.landmark_std)
            feat = (feat - self.landmark_mean) / denom

        return {
            "x": x,
            "landmark": to_1d_tensor(feat),
            "y": torch.tensor(s.label, dtype=torch.long),
        }
