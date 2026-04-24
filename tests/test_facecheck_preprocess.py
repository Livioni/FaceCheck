import numpy as np

from facecheck.data.utils import DepthMinMax
from facecheck.inference.preprocess import FaceCheckPreprocessState, FaceCheckPreprocessor


def test_preprocess_shapes():
    state = FaceCheckPreprocessState(
        depth_minmax=DepthMinMax(vmin=1.0, vmax=2.0),
        landmark_mean=np.zeros((27,), dtype=np.float32),
        landmark_std=np.ones((27,), dtype=np.float32),
    )
    pre = FaceCheckPreprocessor(state=state, landmark_dim=27)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.zeros((480, 640), dtype=np.float32)
    lm = np.zeros((27,), dtype=np.float32)
    x, lmt = pre.preprocess_bgr_depth_landmark(img, depth, lm)
    assert tuple(x.shape) == (4, 224, 224)
    assert tuple(lmt.shape) == (27,)

