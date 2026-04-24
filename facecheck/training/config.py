from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrainConfig:
    dataset_root: str
    output_dir: str = "outputs/facecheck"
    seed: int = 42
    val_ratio: float = 0.1
    test_ratio: float = 0.2

    batch_size: int = 32
    num_workers: int = 4

    lr: float = 1e-4
    weight_decay: float = 0.05

    epochs: int = 100
    patience: int = 10

    backbone: str = "vit_base_patch16_224"
    pretrained_ckpt: Optional[str] = None
    landmark_dim: int = 27
    landmark_hidden: int = 256
    dropout: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
