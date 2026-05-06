import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from facecheck.data.utils import DepthMinMax
from facecheck.inference.preprocess import FaceCheckPreprocessState, FaceCheckPreprocessor
from facecheck.models.vit_facecheck import FaceCheckViTConfig, ViTFaceCheck


def _import_dynaface() -> Tuple[Any, Any, Any]:
    def _try_import() -> Tuple[Any, Any, Any]:
        import dynaface.facial as facial  # type: ignore
        import dynaface.measures as measures  # type: ignore
        from dynaface import models  # type: ignore

        return facial, measures, models

    try:
        return _try_import()
    except Exception:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        lib = os.path.join(root, "dynaface", "dynaface-lib")
        if os.path.isdir(lib) and lib not in sys.path:
            sys.path.insert(0, lib)
        for k in [k for k in list(sys.modules) if k == "dynaface" or k.startswith("dynaface.")]:
            sys.modules.pop(k, None)
        import importlib

        importlib.invalidate_caches()
        return _try_import()


_DYNAFACE_INITED = False


def _ensure_dynaface_models(models: Any, device: Optional[str] = None) -> str:
    global _DYNAFACE_INITED
    dev = models.detect_device() if device is None else device
    model_dir_env = os.environ.get("DYNAFACE_MODEL_DIR", "").strip()
    model_path = models.download_models(model_dir_env or None)
    if not _DYNAFACE_INITED:
        models.init_models(model_path, dev)
        _DYNAFACE_INITED = True
    return dev


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
        dim = int(self.pre.landmark_dim)
        try:
            facial, measures, models = _import_dynaface()
            _ensure_dynaface_models(models)
            analyzer = facial.AnalyzeFace(measures=[measures.AnalyzeLandmarks()])
            ok = analyzer.load_image(img_bgr, crop=False)
            if not ok or analyzer.is_no_face():
                return np.zeros((dim,), dtype=np.float32)
            lm_dict = analyzer.analyze()
            if lm_dict is None:
                return np.zeros((dim,), dtype=np.float32)
            n = int(getattr(measures.AnalyzeLandmarks, "NUM_LANDMARKS", 97))
            xy = np.zeros((n, 2), dtype=np.float32)
            for i in range(1, n + 1):
                xy[i - 1, 0] = float(lm_dict.get(f"landmark-{i}-x", 0.0))
                xy[i - 1, 1] = float(lm_dict.get(f"landmark-{i}-y", 0.0))
        except Exception:
            return np.zeros((dim,), dtype=np.float32)

        h, w = img_bgr.shape[:2]
        xy[:, 0] = xy[:, 0] / max(float(w), 1.0)
        xy[:, 1] = xy[:, 1] / max(float(h), 1.0)
        flat = xy.reshape(-1)
        if flat.size < dim:
            flat = np.concatenate([flat, np.zeros((dim - flat.size,), dtype=np.float32)], axis=0)
        return flat[:dim].astype(np.float32, copy=False)

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
