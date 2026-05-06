from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

import timm

from facecheck.models.layers import BinaryHead, LandmarkMLP, LandmarkTokenConfig


@dataclass(frozen=True)
class FaceCheckViTConfig:
    backbone: str = "vit_base_patch16_224"
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 4
    num_classes: int = 2
    landmark_dim: int = 27
    landmark_hidden: int = 256
    dropout: float = 0.0


def _adapt_patch_embed_weight(w: torch.Tensor, in_chans: int) -> torch.Tensor:
    if w.ndim != 4:
        return w
    if w.shape[1] == in_chans:
        return w
    if w.shape[1] == 3 and in_chans == 4:
        extra = w[:, :3].mean(dim=1, keepdim=True)
        return torch.cat([w, extra], dim=1)
    if w.shape[1] > in_chans:
        return w[:, :in_chans]
    pad = in_chans - w.shape[1]
    extra = w.mean(dim=1, keepdim=True).repeat(1, pad, 1, 1)
    return torch.cat([w, extra], dim=1)


class ViTFaceCheck(nn.Module):
    def __init__(self, cfg: FaceCheckViTConfig, pretrained: bool = False) -> None:
        super().__init__()
        self.cfg = cfg

        vit = timm.create_model(
            cfg.backbone,
            pretrained=pretrained,
            num_classes=0,
            in_chans=cfg.in_chans,
            img_size=cfg.img_size,
        )
        self.vit = vit

        embed_dim = getattr(vit, "embed_dim", None)
        if embed_dim is None:
            raise ValueError("Unsupported backbone: missing embed_dim")
        self.embed_dim = int(embed_dim)

        self.landmark = LandmarkMLP(
            LandmarkTokenConfig(
                in_dim=cfg.landmark_dim,
                embed_dim=self.embed_dim,
                hidden_dim=cfg.landmark_hidden,
                dropout=cfg.dropout,
            )
        )
        self.head = BinaryHead(self.embed_dim, dropout=cfg.dropout)

        self.uses_rope = getattr(vit, "rope", None) is not None
        if self.uses_rope:
            for blk in self.vit.blocks:
                attn = getattr(blk, "attn", None)
                if attn is not None and hasattr(attn, "num_prefix_tokens"):
                    attn.num_prefix_tokens = int(attn.num_prefix_tokens) + 1
        else:
            self._ensure_landmark_pos_embed()

    def _ensure_landmark_pos_embed(self) -> None:
        pe = getattr(self.vit, "pos_embed", None)
        if pe is None:
            return
        if pe.ndim != 3:
            return
        num_patches = getattr(getattr(self.vit, "patch_embed", None), "num_patches", None)
        if num_patches is None:
            return
        expected = int(num_patches) + 2
        if pe.shape[1] == expected:
            return

        old = self.vit.pos_embed
        if old.shape[1] == 0:
            return
        new = nn.Parameter(torch.zeros((1, old.shape[1] + 1, old.shape[2]), dtype=old.dtype))
        with torch.no_grad():
            new[:, : old.shape[1]] = old
        self.vit.pos_embed = new

    def forward_features(self, x: torch.Tensor, landmark: torch.Tensor) -> torch.Tensor:
        if self.uses_rope:
            return self._forward_features_rope(x, landmark)
        return self._forward_features_abs(x, landmark)

    def _forward_features_abs(self, x: torch.Tensor, landmark: torch.Tensor) -> torch.Tensor:
        vit = self.vit
        x = vit.patch_embed(x)
        if isinstance(x, (tuple, list)):
            x = x[0]
        if x.ndim == 4:
            if x.shape[-1] == self.embed_dim:
                x = x.reshape(x.shape[0], -1, x.shape[-1])
            else:
                x = x.flatten(2).transpose(1, 2)
        cls = vit.cls_token.expand(x.shape[0], -1, -1)
        lm = self.landmark(landmark)

        x = torch.cat([cls, x, lm], dim=1)

        pos = vit.pos_embed
        if pos is not None and pos.shape[1] != x.shape[1]:
            if pos.shape[1] == x.shape[1] - 1:
                pad = torch.zeros((1, 1, pos.shape[2]), device=pos.device, dtype=pos.dtype)
                pos = torch.cat([pos, pad], dim=1)
            else:
                pos = None
        if pos is not None:
            x = x + pos

        x = vit.pos_drop(x)
        for blk in vit.blocks:
            x = blk(x)
        x = vit.norm(x)
        return x[:, 0]

    def _forward_features_rope(self, x: torch.Tensor, landmark: torch.Tensor) -> torch.Tensor:
        vit = self.vit
        x = vit.patch_embed(x)
        if isinstance(x, (tuple, list)):
            x = x[0]
        if x.ndim == 4:
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)
            grid = (H, W)
        else:
            B, N, _ = x.shape
            side = int(round(N ** 0.5))
            grid = (side, side)

        rope = getattr(vit, "rope", None)
        if rope is not None:
            if getattr(vit, "dynamic_img_size", False):
                rot_pos_embed = rope.get_embed(shape=grid)
            else:
                rot_pos_embed = rope.get_embed()
        else:
            rot_pos_embed = None

        prefix = []
        if vit.cls_token is not None:
            prefix.append(vit.cls_token.expand(x.shape[0], -1, -1))
        reg = getattr(vit, "reg_token", None)
        if reg is not None:
            prefix.append(reg.expand(x.shape[0], -1, -1))
        prefix.append(self.landmark(landmark))
        x = torch.cat(prefix + [x], dim=1)

        x = vit.pos_drop(x)
        norm_pre = getattr(vit, "norm_pre", None)
        if norm_pre is not None:
            x = norm_pre(x)
        for blk in vit.blocks:
            x = blk(x, rope=rot_pos_embed)
        x = vit.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor, landmark: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x, landmark)
        return self.head(feat)

    def load_pretrained(self, ckpt_path: str, strict: bool = False) -> Dict[str, Any]:
        obj = torch.load(ckpt_path, map_location="cpu")
        state: Dict[str, Any]
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        elif isinstance(obj, dict):
            state = obj
        else:
            raise ValueError("Unsupported checkpoint format")

        adapted: Dict[str, Any] = {}
        for k, v in state.items():
            key = k
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("vit."):
                key2 = key
            else:
                key2 = "vit." + key if key in self.vit.state_dict() else key

            if key2.endswith("patch_embed.proj.weight") and isinstance(v, torch.Tensor):
                v = _adapt_patch_embed_weight(v, self.cfg.in_chans)
            adapted[key2] = v

        missing, unexpected = self.load_state_dict(adapted, strict=strict)
        return {"missing_keys": missing, "unexpected_keys": unexpected}

    def load_timm_pretrained(self, backbone_name: Optional[str] = None) -> Dict[str, Any]:
        name = backbone_name or self.cfg.backbone
        ref = timm.create_model(
            name,
            pretrained=True,
            num_classes=0,
            in_chans=3,
            img_size=self.cfg.img_size,
        )
        sd = ref.state_dict()

        adapted: Dict[str, Any] = {}
        for k, v in sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k == "pos_embed":
                pe = v
                cur = getattr(self.vit, "pos_embed", None)
                if cur is not None and isinstance(cur, torch.Tensor) and cur.ndim == 3:
                    if pe.shape[1] == cur.shape[1] - 1:
                        pad = torch.zeros((1, 1, pe.shape[2]), dtype=pe.dtype)
                        pe = torch.cat([pe, pad], dim=1)
                v = pe
            if k == "patch_embed.proj.weight":
                v = _adapt_patch_embed_weight(v, self.cfg.in_chans)
            adapted["vit." + k] = v

        missing, unexpected = self.load_state_dict(adapted, strict=False)
        return {"missing_keys": missing, "unexpected_keys": unexpected}
