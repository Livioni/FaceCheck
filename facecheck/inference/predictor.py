import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from facecheck.data.utils import DepthMinMax
from facecheck.inference.preprocess import FaceCheckPreprocessState, FaceCheckPreprocessor
from facecheck.models.vit_facecheck import FaceCheckViTConfig, ViTFaceCheck


def _import_dynaface() -> Any:
    try:
        import dynaface.facial as facial  # type: ignore

        return facial
    except Exception:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        lib = os.path.join(root, "dynaface", "dynaface-lib")
        if os.path.isdir(lib) and lib not in sys.path:
            sys.path.insert(0, lib)
        import dynaface.facial as facial  # type: ignore

        return facial


@dataclass(frozen=True)
class PredictResult:
    prob_affected: float
    label: str


class FaceCheckPredictor:
    def __init__(self, model: ViTFaceCheck, pre: FaceCheckPreprocessor, device: torch.device) -> None:
        self.model = model.eval()
        self.pre = pre
        self.device = device
        self.model.to(self.device)

    @staticmethod
    def load(ckpt_path: str, device: Optional[str] = None) -> "FaceCheckPredictor":
        ckpt = torch.load(ckpt_path, map_location="cpu")
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

        state = FaceCheckPreprocessState(
            depth_minmax=DepthMinMax(vmin=float(depth_minmax["vmin"]), vmax=float(depth_minmax["vmax"])),
            landmark_mean=None if lm_mean is None else np.asarray(lm_mean, dtype=np.float32),
            landmark_std=None if lm_std is None else np.asarray(lm_std, dtype=np.float32),
        )
        pre = FaceCheckPreprocessor(state=state, landmark_dim=int(cfg_dict.get("landmark_dim", 27)))

        dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return FaceCheckPredictor(model=model, pre=pre, device=dev)

    def _dynaface_landmark_vec(self, img_bgr: np.ndarray) -> np.ndarray:
        facial = _import_dynaface()
        lms = facial.infer_landmarks_bgr(img_bgr, crop=False)
        if not lms:
            return np.zeros((self.pre.landmark_dim,), dtype=np.float32)
        h, w = img_bgr.shape[:2]
        arr = np.asarray(lms, dtype=np.float32)
        arr[:, 0] = arr[:, 0] / max(float(w), 1.0)
        arr[:, 1] = arr[:, 1] / max(float(h), 1.0)
        flat = arr.reshape(-1)
        if flat.size < self.pre.landmark_dim:
            pad = np.zeros((self.pre.landmark_dim - flat.size,), dtype=np.float32)
            flat = np.concatenate([flat, pad], axis=0)
        return flat[: self.pre.landmark_dim]

    @torch.no_grad()
    def predict_bgr_depth(
        self,
        img_bgr: np.ndarray,
        depth: np.ndarray,
        landmark_vec: Optional[np.ndarray] = None,
    ) -> PredictResult:
        if landmark_vec is None:
            landmark_vec = self._dynaface_landmark_vec(img_bgr)
        x, lm = self.pre.preprocess_bgr_depth_landmark(img_bgr, depth, landmark_vec)
        x = x.unsqueeze(0).to(self.device)
        lm = lm.unsqueeze(0).to(self.device)
        logits = self.model(x, lm)
        prob = float(torch.softmax(logits, dim=1)[0, 1].detach().cpu().item())
        label = "affected" if prob >= 0.5 else "unaffected"
        return PredictResult(prob_affected=prob, label=label)

    @torch.no_grad()
    def predict_paths(
        self,
        rgb_path: str,
        depth_path: str,
        landmark_vec: Optional[np.ndarray] = None,
    ) -> PredictResult:
        import cv2

        img_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(rgb_path)
        depth = np.load(depth_path)
        return self.predict_bgr_depth(img_bgr, depth, landmark_vec=landmark_vec)
