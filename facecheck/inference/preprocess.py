from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from facecheck.data.utils import DepthMinMax, read_depth_224, to_1d_tensor, to_4ch_tensor


@dataclass(frozen=True)
class FaceCheckPreprocessState:
    depth_minmax: DepthMinMax
    landmark_mean: Optional[np.ndarray]
    landmark_std: Optional[np.ndarray]


class FaceCheckPreprocessor:
    def __init__(self, state: FaceCheckPreprocessState, landmark_dim: int = 27) -> None:
        self.state = state
        self.landmark_dim = int(landmark_dim)

    def preprocess_bgr_depth_landmark(
        self,
        img_bgr: np.ndarray,
        depth: np.ndarray,
        landmark_vec: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Empty image")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        rgb_01 = img_rgb.astype(np.float32) / 255.0

        if depth.ndim == 3:
            depth = depth.squeeze()
        if depth.ndim != 2:
            raise ValueError(f"Invalid depth shape: {depth.shape}")
        depth = cv2.resize(depth.astype(np.float32), (224, 224), interpolation=cv2.INTER_NEAREST)
        depth_01 = self.state.depth_minmax.normalize(depth)

        lm = np.asarray(landmark_vec, dtype=np.float32).reshape(-1)
        if lm.size != self.landmark_dim:
            raise ValueError(f"Invalid landmark dim: {lm.size} != {self.landmark_dim}")
        if self.state.landmark_mean is not None and self.state.landmark_std is not None:
            denom = np.where(self.state.landmark_std == 0, 1.0, self.state.landmark_std)
            lm = (lm - self.state.landmark_mean) / denom

        return to_4ch_tensor(rgb_01, depth_01), to_1d_tensor(lm)

    def preprocess_paths(
        self,
        rgb_path: str,
        depth_path: str,
        landmark_vec: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(rgb_path)
        depth = read_depth_224(depth_path)
        return self.preprocess_bgr_depth_landmark(img_bgr, depth, landmark_vec)

