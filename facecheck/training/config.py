from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrainConfig:
    dataset_root: str
    output_dir: str = "outputs/facecheck"
    seed: int = 42
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    human_faces_root: Optional[str] = None
    human_faces_csv: Optional[str] = None
    human_faces_max_train: int = 0
    human_faces_max_val: int = 0
    human_faces_max_test: int = 0
    human_faces_in_val_test: bool = False

    batch_size: int = 32
    num_workers: int = 4

    lr: float = 1e-4
    backbone_lr_mult: float = 0.1
    weight_decay: float = 0.1
    warmup_epochs: int = 3
    min_lr_ratio: float = 0.01
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    use_class_weights: bool = True

    epochs: int = 100
    patience: int = 10

    backbone: str = "vit_base_patch16_dinov3"
    pretrained_ckpt: Optional[str] = None
    landmark_dim: int = 30
    landmark_hidden: int = 256
    dropout: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
