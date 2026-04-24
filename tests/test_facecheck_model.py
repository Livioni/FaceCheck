import torch

from facecheck.models.vit_facecheck import FaceCheckViTConfig, ViTFaceCheck


def test_vit_facecheck_forward_shape():
    cfg = FaceCheckViTConfig(landmark_dim=27)
    model = ViTFaceCheck(cfg, pretrained=False)
    x = torch.zeros((2, 4, 224, 224), dtype=torch.float32)
    lm = torch.zeros((2, 27), dtype=torch.float32)
    y = model(x, lm)
    assert y.shape == (2, 2)

