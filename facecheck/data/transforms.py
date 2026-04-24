from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    enable: bool = True
    hflip_p: float = 0.5
    rotate_deg: float = 10.0
    color_jitter_p: float = 0.8
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    depth_jitter_p: float = 0.5
    depth_scale: float = 0.1
    depth_bias: float = 0.05
    depth_noise: float = 0.01


def _maybe_hflip(rng: np.random.Generator, rgb: np.ndarray, depth: np.ndarray, p: float) -> Tuple[np.ndarray, np.ndarray]:
    if rng.random() >= p:
        return rgb, depth
    rgb = cv2.flip(rgb, 1)
    depth = cv2.flip(depth, 1)
    return rgb, depth


def _maybe_rotate(rng: np.random.Generator, rgb: np.ndarray, depth: np.ndarray, deg: float) -> Tuple[np.ndarray, np.ndarray]:
    if deg <= 0:
        return rgb, depth
    angle = float(rng.uniform(-deg, deg))
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rgb2 = cv2.warpAffine(rgb, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    depth2 = cv2.warpAffine(depth, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rgb2, depth2


def _color_jitter(rgb: np.ndarray, brightness: float, contrast: float, saturation: float, rng: np.random.Generator) -> np.ndarray:
    out = rgb.astype(np.float32, copy=True)
    ops = ["brightness", "contrast", "saturation"]
    rng.shuffle(ops)

    for op in ops:
        if op == "brightness" and brightness > 0:
            delta = float(rng.uniform(-brightness, brightness))
            out = out + delta
        elif op == "contrast" and contrast > 0:
            factor = float(rng.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast))
            mean = float(np.mean(out))
            out = (out - mean) * factor + mean
        elif op == "saturation" and saturation > 0:
            factor = float(rng.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation))
            gray = np.dot(out, np.array([0.299, 0.587, 0.114], dtype=np.float32))
            gray3 = gray[..., None]
            out = gray3 + factor * (out - gray3)

    return np.clip(out, 0.0, 1.0)


def _maybe_color_jitter(rng: np.random.Generator, rgb: np.ndarray, cfg: AugmentConfig) -> np.ndarray:
    if cfg.color_jitter_p <= 0 or rng.random() >= cfg.color_jitter_p:
        return rgb
    if cfg.brightness <= 0 and cfg.contrast <= 0 and cfg.saturation <= 0:
        return rgb
    return _color_jitter(rgb, cfg.brightness, cfg.contrast, cfg.saturation, rng)


def _maybe_depth_jitter(rng: np.random.Generator, depth: np.ndarray, cfg: AugmentConfig) -> np.ndarray:
    if cfg.depth_jitter_p <= 0 or rng.random() >= cfg.depth_jitter_p:
        return depth
    out = depth.astype(np.float32, copy=True)
    mask = out > 0
    if not np.any(mask):
        return out

    if cfg.depth_scale > 0:
        scale = float(rng.uniform(max(0.0, 1.0 - cfg.depth_scale), 1.0 + cfg.depth_scale))
        out[mask] = out[mask] * scale
    if cfg.depth_bias != 0:
        bias = float(rng.uniform(-cfg.depth_bias, cfg.depth_bias))
        out[mask] = out[mask] + bias
    if cfg.depth_noise > 0:
        noise = rng.normal(loc=0.0, scale=float(cfg.depth_noise), size=out.shape).astype(np.float32)
        out[mask] = out[mask] + noise[mask]

    out = np.maximum(out, 0.0)
    return out


def augment_pair(
    rgb: np.ndarray,
    depth: np.ndarray,
    cfg: AugmentConfig,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not cfg.enable:
        return rgb, depth
    rng = np.random.default_rng(seed)
    rgb, depth = _maybe_hflip(rng, rgb, depth, cfg.hflip_p)
    rgb, depth = _maybe_rotate(rng, rgb, depth, cfg.rotate_deg)
    rgb = _maybe_color_jitter(rng, rgb, cfg)
    depth = _maybe_depth_jitter(rng, depth, cfg)
    return rgb, depth
