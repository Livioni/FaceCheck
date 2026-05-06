import argparse
import glob
import json
import os
import sys
import warnings
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from skimage.transform import SimilarityTransform, estimate_transform, warp


@dataclass(frozen=True)
class CropResult:
    tform_params: np.ndarray
    cropped_bgr_pre_resize: np.ndarray
    cropped_bgr_224: np.ndarray
    cropped_bgr_no_resize: np.ndarray
    crop_bbox_xyxy: Tuple[int, int, int, int]
    mediapipe_xy: np.ndarray
    cropped_mediapipe_xy: np.ndarray


def _repo_root() -> str:
    return os.path.abspath(os.path.dirname(__file__))


def _pushd(path: str) -> str:
    old = os.getcwd()
    os.chdir(path)
    return old


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _write_json(path: str, obj: Any) -> None:
    def _default(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)


def _resolve_existing_file(path: str, bases: Tuple[str, ...]) -> str:
    p = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(p):
        ap = os.path.abspath(p)
        if os.path.isfile(ap):
            return ap
        raise FileNotFoundError(ap)

    tried = []
    tried_set = set()
    for b in bases:
        cand = os.path.abspath(os.path.join(b, p))
        if cand not in tried_set:
            tried.append(cand)
            tried_set.add(cand)
        if os.path.isfile(cand):
            return cand

    raise FileNotFoundError(f"{path} (tried: {', '.join(tried)})")


def _discover_facecheck_ckpt(bases: Tuple[str, ...]) -> Tuple[str, ...]:
    pats = (
        os.path.join("outputs", "facecheck", "best.pt"),
        os.path.join("outputs", "**", "best.pt"),
        os.path.join("outputs", "**", "*.pt"),
        os.path.join("outputs", "**", "*.pth"),
        os.path.join("checkpoints", "**", "*.pt"),
        os.path.join("checkpoints", "**", "*.pth"),
    )
    out = []
    seen = set()
    for b in bases:
        for pat in pats:
            for p in glob.glob(os.path.join(b, pat), recursive=True):
                ap = os.path.abspath(p)
                if os.path.isfile(ap) and ap not in seen:
                    out.append(ap)
                    seen.add(ap)
    out.sort()
    return tuple(out)


def _resolve_facecheck_ckpt(path: str, bases: Tuple[str, ...]) -> str:
    try:
        return _resolve_existing_file(path, bases)
    except FileNotFoundError as e:
        cands = _discover_facecheck_ckpt(bases)
        if len(cands) == 1:
            return cands[0]
        msg = str(e)
        if cands:
            msg = f"{msg}\n已在常见目录找到可用 ckpt（请用 --facecheck_ckpt 显式指定其中一个）：\n- " + "\n- ".join(cands)
        else:
            msg = f"{msg}\n未找到任何 FaceCheck ckpt。请先训练生成 outputs/facecheck/best.pt，或把现成 ckpt 的绝对路径传给 --facecheck_ckpt。"
        raise FileNotFoundError(msg) from None


def _smirk_dir() -> str:
    root = _repo_root()
    smirk_dir = os.path.join(root, "smirk")
    if not os.path.isdir(smirk_dir):
        raise FileNotFoundError(smirk_dir)
    if smirk_dir not in sys.path:
        sys.path.insert(0, smirk_dir)
    return smirk_dir


def _import_smirk() -> Tuple[Any, Any, Any, Any, Any, Any]:
    _smirk_dir()
    from src.smirk_encoder import SmirkEncoder  # type: ignore
    from src.FLAME.FLAME import FLAME  # type: ignore
    from src.renderer.util import batch_orth_proj  # type: ignore
    from src.renderer.renderer import Renderer  # type: ignore
    from utils.mediapipe_utils import run_mediapipe  # type: ignore

    from pytorch3d.structures import Meshes  # type: ignore
    from pytorch3d.renderer.mesh import rasterize_meshes  # type: ignore

    return SmirkEncoder, FLAME, batch_orth_proj, run_mediapipe, (Meshes, rasterize_meshes), Renderer


def _import_dynaface() -> Tuple[Any, Any, Any]:
    root = _repo_root()
    lib = os.path.join(root, "dynaface", "dynaface-lib")
    if os.path.isdir(lib) and lib not in sys.path:
        sys.path.insert(0, lib)
    import dynaface.facial as facial  # type: ignore
    import dynaface.measures as measures  # type: ignore
    from dynaface import models  # type: ignore

    return facial, measures, models


def crop_like_smirk_demo(
    img_bgr: np.ndarray,
    image_size: int = 224,
    scale: float = 1.4,
) -> CropResult:
    SmirkEncoder, FLAME, batch_orth_proj, run_mediapipe, _, _ = _import_smirk()
    _ = SmirkEncoder, FLAME, batch_orth_proj
    kpt = run_mediapipe(img_bgr)
    if kpt is None:
        raise RuntimeError("mediapipe 未检测到人脸关键点，无法 crop")
    kpt_xy = np.asarray(kpt[..., :2], dtype=np.float32)

    left = float(np.min(kpt_xy[:, 0]))
    right = float(np.max(kpt_xy[:, 0]))
    top = float(np.min(kpt_xy[:, 1]))
    bottom = float(np.max(kpt_xy[:, 1]))
    old_size = (right - left + bottom - top) / 2.0
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0], dtype=np.float32)
    size = int(old_size * scale)

    src_pts = np.array(
        [
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2],
        ],
        dtype=np.float32,
    )
    dst_pts = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]], dtype=np.float32)
    tform = estimate_transform("similarity", src_pts, dst_pts)

    cropped_pre_resize = warp(img_bgr, tform.inverse, output_shape=(image_size, image_size), preserve_range=True).astype(
        np.uint8
    )
    cropped_kpt = (np.dot(tform.params, np.hstack([kpt_xy, np.ones([kpt_xy.shape[0], 1], dtype=np.float32)]).T).T)[
        :, :2
    ].astype(np.float32)

    cropped = cv2.resize(cropped_pre_resize, (image_size, image_size), interpolation=cv2.INTER_AREA)

    x1 = int(round(center[0] - size / 2.0))
    y1 = int(round(center[1] - size / 2.0))
    x2 = x1 + size
    y2 = y1 + size
    H, W = img_bgr.shape[:2]
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - W)
    pad_bottom = max(0, y2 - H)
    if pad_left or pad_top or pad_right or pad_bottom:
        padded = cv2.copyMakeBorder(
            img_bgr, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
        cropped_no_resize = padded[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left].copy()
    else:
        cropped_no_resize = img_bgr[y1:y2, x1:x2].copy()

    return CropResult(
        tform_params=np.asarray(tform.params, dtype=np.float32),
        cropped_bgr_pre_resize=cropped_pre_resize,
        cropped_bgr_224=cropped,
        cropped_bgr_no_resize=cropped_no_resize,
        crop_bbox_xyxy=(x1, y1, x2, y2),
        mediapipe_xy=kpt_xy,
        cropped_mediapipe_xy=cropped_kpt,
    )


@torch.no_grad()
def smirk_depth_from_cropped_bgr_224(
    cropped_bgr_224: np.ndarray,
    smirk_ckpt: str,
    device: str,
) -> Dict[str, Any]:
    SmirkEncoder, FLAME, batch_orth_proj, _, (Meshes, rasterize_meshes), Renderer = _import_smirk()

    smirk_dir = _smirk_dir()
    smirk_ckpt_abs = _resolve_existing_file(smirk_ckpt, (os.getcwd(), _repo_root(), smirk_dir))

    dev = torch.device(device)
    old = _pushd(_repo_root())
    try:
        smirk_encoder = SmirkEncoder().to(dev).eval()
        ckpt = torch.load(smirk_ckpt_abs, map_location="cpu")
        encoder_state = {k.replace("smirk_encoder.", ""): v for k, v in ckpt.items() if "smirk_encoder" in k}
        smirk_encoder.load_state_dict(encoder_state, strict=True)

        flame = FLAME().to(dev).eval()
        renderer = Renderer().to(dev).eval()

        rgb = cv2.cvtColor(cropped_bgr_224, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = x.to(dev)

        outputs = smirk_encoder(x)
        flame_out = flame.forward(outputs)
        vertices = flame_out["vertices"]
        faces = flame.faces_tensor.unsqueeze(0).expand(vertices.shape[0], -1, -1)

        renderer_out = renderer.forward(
            vertices,
            outputs["cam"],
            landmarks_fan=flame_out.get("landmarks_fan"),
            landmarks_mp=flame_out.get("landmarks_mp"),
        )
        rendered_img_t = renderer_out["rendered_img"].clamp(0.0, 1.0)
        rendered_rgb = (rendered_img_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.addWeighted(cropped_bgr_224, 0.5, rendered_bgr, 0.5, 0.0)

        transformed_vertices = batch_orth_proj(vertices, outputs["cam"])
        transformed_vertices[:, :, 1:] = -transformed_vertices[:, :, 1:]
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10.0
        fixed_vertices = transformed_vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]

        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        pix_to_face, zbuf, _, _ = rasterize_meshes(
            meshes_screen,
            image_size=224,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None,
            perspective_correct=False,
        )
        mask = (pix_to_face[..., 0] > -1).detach().cpu().numpy().astype(np.uint8)
        depth = zbuf[..., 0].detach().cpu().numpy().astype(np.float32)
        depth = np.where(mask > 0, depth, 0.0).astype(np.float32)

        return {
            "depth": depth,
            "mask": mask,
            "mesh_bgr": rendered_bgr,
            "mesh_overlay_bgr": overlay_bgr,
            "smirk_outputs": {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in outputs.items()},
            "flame": {
                "vertices": vertices.detach().cpu().numpy(),
                "landmarks_mp": flame_out.get("landmarks_mp").detach().cpu().numpy() if "landmarks_mp" in flame_out else None,
                "landmarks_fan": flame_out.get("landmarks_fan").detach().cpu().numpy() if "landmarks_fan" in flame_out else None,
            },
        }
    finally:
        os.chdir(old)


def _depth_to_vis(depth: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    d = np.asarray(depth)
    if d.ndim == 3 and d.shape[0] == 1:
        d = d[0]
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    d = d.astype(np.float32)

    m: Optional[np.ndarray] = None
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        if m.ndim == 3 and m.shape[-1] == 1:
            m = m[..., 0]

    if m is not None:
        valid = (m > 0) & np.isfinite(d)
    else:
        valid = np.isfinite(d)

    if not np.any(valid):
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)

    v = d[valid]
    vmin = float(np.percentile(v, 1))
    vmax = float(np.percentile(v, 99))
    denom = max(vmax - vmin, 1e-6)

    gray = np.clip((d - vmin) / denom, 0.0, 1.0)
    gray_u8 = np.ascontiguousarray((gray * 255.0).round().astype(np.uint8))
    colored = cv2.applyColorMap(gray_u8, cv2.COLORMAP_TURBO)

    if m is not None:
        colored = np.where(m[..., None] > 0, colored, 0)

    return colored


def dynaface_landmarks_and_overlay(img_bgr: np.ndarray, device: Optional[str] = None) -> Dict[str, Any]:
    facial, measures, models = _import_dynaface()
    dev = models.detect_device() if device is None else device
    model_dir_env = os.environ.get("DYNAFACE_MODEL_DIR", "").strip()
    model_path = models.download_models(model_dir_env or None)
    models.init_models(model_path, dev)

    analyzer = facial.AnalyzeFace(measures=[measures.AnalyzeLandmarks()])
    ok = analyzer.load_image(img_bgr, crop=False)
    if not ok or analyzer.is_no_face():
        return {
            "ok": False,
            "device": dev,
            "model_path": model_path,
            "landmarks_xy": None,
            "landmarks_dict": None,
            "overlay_bgr": None,
            "overlay_numbers_bgr": None,
        }

    lm_dict = analyzer.analyze()
    if lm_dict is None:
        return {
            "ok": False,
            "device": dev,
            "model_path": model_path,
            "landmarks_xy": None,
            "landmarks_dict": None,
            "overlay_bgr": None,
            "overlay_numbers_bgr": None,
        }

    n = int(getattr(measures.AnalyzeLandmarks, "NUM_LANDMARKS", 97))
    xy = np.zeros((n, 2), dtype=np.float32)
    for i in range(1, n + 1):
        xy[i - 1, 0] = float(lm_dict.get(f"landmark-{i}-x", 0.0))
        xy[i - 1, 1] = float(lm_dict.get(f"landmark-{i}-y", 0.0))

    overlay = analyzer.render_img.copy()
    analyzer.draw_landmarks(numbers=True)
    overlay_numbers = analyzer.render_img.copy()
    return {
        "ok": True,
        "device": dev,
        "model_path": model_path,
        "landmarks_xy": xy,
        "landmarks_dict": lm_dict,
        "overlay_bgr": overlay,
        "overlay_numbers_bgr": overlay_numbers,
    }


def dynaface_quicktest_save(img_bgr: np.ndarray, out_path: str, device: Optional[str] = None) -> Dict[str, Any]:
    facial, measures, models = _import_dynaface()
    dev = models.detect_device() if device is None else device
    model_dir_env = os.environ.get("DYNAFACE_MODEL_DIR", "").strip()
    model_path = models.download_models(model_dir_env or None)
    models.init_models(model_path, dev)

    analyzer = facial.AnalyzeFace(measures=measures.all_measures())
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ok = analyzer.load_image(img_rgb, crop=False)
    if not ok or analyzer.is_no_face():
        return {"ok": False, "device": dev, "model_path": model_path, "save_path": None}

    lm_dict = analyzer.analyze()
    if lm_dict is None:
        return {"ok": False, "device": dev, "model_path": model_path, "save_path": None}

    analyzer.draw_landmarks(numbers=True)
    analyzer.save(out_path)
    return {"ok": True, "device": dev, "model_path": model_path, "save_path": out_path, "measures": lm_dict}


def _landmark_vec_for_facecheck(predictor: Any, img_bgr: np.ndarray, lms_xy: Optional[np.ndarray]) -> np.ndarray:
    dim = int(predictor.pre.landmark_dim)
    if lms_xy is None or lms_xy.size == 0:
        return np.zeros((dim,), dtype=np.float32)
    h, w = img_bgr.shape[:2]
    arr = np.asarray(lms_xy, dtype=np.float32).copy()
    arr[:, 0] = arr[:, 0] / max(float(w), 1.0)
    arr[:, 1] = arr[:, 1] / max(float(h), 1.0)
    flat = arr.reshape(-1)
    if flat.size < dim:
        flat = np.concatenate([flat, np.zeros((dim - flat.size,), dtype=np.float32)], axis=0)
    return flat[:dim]


def _put_text(img_bgr: np.ndarray, lines: Tuple[str, ...]) -> np.ndarray:
    canvas = img_bgr.copy()
    h, w = canvas.shape[:2]
    x = 10
    y = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_scale = 0.65
    line_gap = 6
    for ln in lines:
        font_scale = base_scale
        max_w = max(w - x * 2, 1)
        (tw, th), baseline = cv2.getTextSize(ln, font, font_scale, 1)
        if tw > max_w:
            font_scale = max(0.35, font_scale * (max_w / max(tw, 1)))
            (tw, th), baseline = cv2.getTextSize(ln, font, font_scale, 1)

        y_text = min(y + th, h - 1)
        cv2.putText(canvas, ln, (x, y_text), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, ln, (x, y_text), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        y = y + th + baseline + line_gap
    return canvas


def _hstack(images: Tuple[np.ndarray, ...], height: int) -> np.ndarray:
    resized = []
    for im in images:
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, axis=2)
        if im.shape[2] == 4:
            im = im[:, :, :3]
        h, w = im.shape[:2]
        nw = int(round(w * (height / max(h, 1))))
        resized.append(cv2.resize(im, (nw, height), interpolation=cv2.INTER_AREA))
    return np.concatenate(resized, axis=1)


_HEIC_DEPTH_PRIORITY = ("depth", "disparity", "hdrgainmap")
_HEIC_MATTE_PRIORITY = ("portraiteffectsmatte", "semanticskinmatte", "semanticskymatte")


def _heif_aux_to_array(aux_img: Any) -> np.ndarray:
    """Convert pillow_heif aux image → float32 H×W via raw frombytes (handles 10/16-bit)."""
    from PIL import Image  # type: ignore
    pil = Image.frombytes(aux_img.mode, aux_img.size, aux_img.data, "raw", aux_img.mode, aux_img.stride)
    arr = np.asarray(pil, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def load_heic_rgb_and_depth(
    heic_path: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load iPhone HEIC in a single pass (follows finnschi/heic-shenanigans approach).

    Returns (bgr, depth_or_None, portrait_matte_or_None).

    depth priority:
      1. info['depth_images']  — true geometric depth (LiDAR / TrueDepth)
      2. aux 'depth' / 'disparity'
      3. aux 'hdrgainmap'      — best spatial proxy when no true depth available

    portrait_matte: 'portraiteffectsmatte' / 'semanticskinmatte' — used for face-region masking.
    All arrays are float32 H×W in native (non-normalised) units.
    """
    try:
        import pillow_heif as _pheif  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        raise RuntimeError("无法读取 HEIC，请安装 pillow-heif：pip install pillow-heif")

    heif = _pheif.read_heif(heic_path)

    # --- RGB ---
    rgb_pil = Image.frombytes(heif.mode, heif.size, heif.data, "raw", heif.mode, heif.stride)
    bgr = cv2.cvtColor(np.asarray(rgb_pil.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2BGR)

    # --- collect aux by keyword ---
    aux_map: Dict[str, Any] = heif.info.get("aux", {})
    depth_candidates: list[tuple[int, str, int]] = []
    matte_candidates: list[tuple[int, str, int]] = []
    for urn, ids in aux_map.items():
        urn_lower = urn.lower()
        for prio, key in enumerate(_HEIC_DEPTH_PRIORITY):
            if key in urn_lower:
                for img_id in ids:
                    depth_candidates.append((prio, urn, img_id))
                break
        for prio, key in enumerate(_HEIC_MATTE_PRIORITY):
            if key in urn_lower:
                for img_id in ids:
                    matte_candidates.append((prio, urn, img_id))
                break
    depth_candidates.sort(key=lambda t: t[0])
    matte_candidates.sort(key=lambda t: t[0])

    # --- depth: true depth_images first, then aux ---
    depth: Optional[np.ndarray] = None
    for i, di in enumerate(heif.info.get("depth_images", [])):
        try:
            pil = Image.frombytes(di.mode, di.size, di.data, "raw", di.mode, di.stride)
            arr = np.asarray(pil, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[..., 0]
            print(f"  [HEIC depth] depth_images[{i}] mode={di.mode} shape={arr.shape} "
                  f"range=[{arr.min():.2f},{arr.max():.2f}]")
            depth = arr
            break
        except Exception:
            continue

    if depth is None:
        for _, urn, img_id in depth_candidates:
            try:
                depth = _heif_aux_to_array(heif.get_aux_image(img_id))
                print(f"  [HEIC depth] aux={urn.split(':')[-1]} id={img_id} "
                      f"shape={depth.shape} range=[{depth.min():.2f},{depth.max():.2f}]")
                break
            except Exception:
                continue

    # --- portrait matte ---
    matte: Optional[np.ndarray] = None
    for _, urn, img_id in matte_candidates:
        try:
            matte = _heif_aux_to_array(heif.get_aux_image(img_id))
            print(f"  [HEIC matte] aux={urn.split(':')[-1]} id={img_id} "
                  f"shape={matte.shape} range=[{matte.min():.2f},{matte.max():.2f}]")
            break
        except Exception:
            continue

    return bgr, depth, matte


def face_mask_from_landmarks(kpt_xy: np.ndarray, image_size: int = 224) -> np.ndarray:
    """Filled convex hull of face landmarks → binary uint8 H×W mask."""
    pts = np.clip(np.asarray(kpt_xy, dtype=np.float32), 0, image_size - 1).astype(np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask


def normalize_depth_minmax(depth: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Min-max normalize depth to [0, 1] over valid pixels only."""
    d = depth.astype(np.float32)
    if mask is not None:
        valid = (mask > 0) & np.isfinite(d)
    else:
        valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.zeros_like(d)
    vmin = float(np.min(d[valid]))
    vmax = float(np.max(d[valid]))
    denom = max(vmax - vmin, 1e-6)
    out = np.zeros_like(d)
    out[valid] = np.clip((d[valid] - vmin) / denom, 0.0, 1.0)
    return out


def crop_depth_like_smirk(
    depth: np.ndarray,
    tform_params: np.ndarray,
    image_size: int = 224,
) -> np.ndarray:
    """Apply the same similarity crop transform as crop_like_smirk_demo to a depth map."""
    tform = SimilarityTransform(matrix=tform_params.astype(np.float64))
    cropped = warp(
        depth.astype(np.float64),
        tform.inverse,
        output_shape=(image_size, image_size),
        preserve_range=True,
        order=1,
        mode="constant",
        cval=0.0,
    )
    return cropped.astype(np.float32)


def _print_sep() -> None:
    print("=" * 88)


def _print_stage(title: str) -> None:
    _print_sep()
    print(title)
    _print_sep()


def _print_stage_done(outputs: Tuple[str, ...]) -> None:
    print("输出位置：")
    for p in outputs:
        print(f"  {p}")
    _print_sep()


def run_pipeline(
    *,
    img_bgr: np.ndarray,
    out_dir: str,
    smirk_ckpt: Optional[str],
    facecheck_ckpt: str,
    device: str = "cuda",
    verbose: bool = True,
    depth_override: Optional[np.ndarray] = None,
    face_matte_override: Optional[np.ndarray] = None,
) -> Tuple[str, ...]:
    out_dir = _ensure_dir(os.path.abspath(out_dir))
    outputs: list[str] = []

    if verbose:
        _print_stage("Stage 0 | 初始化")
        _print_stage_done((out_dir,))

    if verbose:
        _print_stage("Stage 1 | 读取输入并保存原图")
    p_input_png = os.path.join(out_dir, "00_input.png")
    cv2.imwrite(p_input_png, img_bgr)
    outputs.append(p_input_png)
    if verbose:
        _print_stage_done((p_input_png,))

    if verbose:
        _print_stage("Stage 2 | Crop（对齐 smirk demo）")
    crop_res = crop_like_smirk_demo(img_bgr, image_size=224, scale=1.4)
    p_cropped_png = os.path.join(out_dir, "01_cropped_224.png")
    cv2.imwrite(p_cropped_png, crop_res.cropped_bgr_224)
    p_crop_json = os.path.join(out_dir, "01_crop.json")
    _write_json(
        p_crop_json,
        {
            "tform_params": crop_res.tform_params,
            "mediapipe_xy": crop_res.mediapipe_xy,
            "cropped_mediapipe_xy": crop_res.cropped_mediapipe_xy,
        },
    )
    outputs.extend([p_cropped_png, p_crop_json])
    if verbose:
        _print_stage_done((p_cropped_png, p_crop_json))

    if verbose:
        _print_stage("Stage 2.1 | DynaFace quick_test（仅 crop 不 resize，原分辨率）")
    p_dynaface_quicktest_jpg = os.path.join(out_dir, "01_dynaface_quicktest.jpg")
    quick_res = dynaface_quicktest_save(crop_res.cropped_bgr_no_resize, p_dynaface_quicktest_jpg, device=None)
    if quick_res.get("save_path"):
        outputs.append(p_dynaface_quicktest_jpg)
        if verbose:
            _print_stage_done((p_dynaface_quicktest_jpg,))
    else:
        if verbose:
            _print_stage_done(tuple())

    if depth_override is not None:
        if verbose:
            _print_stage("Stage 3 | HEIC 深度（裁剪 + 面部过滤 + 归一化 + 可视化）")
        h, w = img_bgr.shape[:2]

        # 1. resize depth to full image resolution, then apply crop
        dshape = depth_override.shape[:2] if depth_override.ndim >= 2 else (depth_override.shape[0],)
        heic_depth_full = (
            depth_override if tuple(dshape) == (h, w)
            else cv2.resize(depth_override, (w, h), interpolation=cv2.INTER_LINEAR)
        )
        heic_depth_cropped = crop_depth_like_smirk(heic_depth_full, crop_res.tform_params, image_size=224)

        # 2. build face mask (priority: portrait matte > mediapipe convex hull > no mask)
        face_mask: Optional[np.ndarray] = None
        mask_source = "none"

        if face_matte_override is not None:
            mh, mw = face_matte_override.shape[:2]
            matte_full = (
                face_matte_override if (mh, mw) == (h, w)
                else cv2.resize(face_matte_override, (w, h), interpolation=cv2.INTER_LINEAR)
            )
            matte_cropped = crop_depth_like_smirk(matte_full, crop_res.tform_params, image_size=224)
            face_mask = (matte_cropped > 128).astype(np.uint8)
            mask_source = "portraiteffectsmatte"

        if face_mask is None and crop_res.cropped_mediapipe_xy is not None and len(crop_res.cropped_mediapipe_xy) >= 3:
            face_mask = face_mask_from_landmarks(crop_res.cropped_mediapipe_xy, image_size=224)
            mask_source = "mediapipe_convex_hull"

        if face_mask is None:
            face_mask = (heic_depth_cropped > 0).astype(np.uint8)
            mask_source = "depth_nonzero"

        if verbose:
            print(f"  面部遮罩来源: {mask_source}  有效像素: {int(face_mask.sum())}")

        # 3. zero out non-face pixels, normalize [0,1]
        heic_depth_cropped = np.where(face_mask > 0, heic_depth_cropped, 0.0).astype(np.float32)
        mask = face_mask
        depth = normalize_depth_minmax(heic_depth_cropped, mask=mask)

        p_depth_npy = os.path.join(out_dir, "02_heic_depth.npy")
        p_mask_npy = os.path.join(out_dir, "02_heic_depth_mask.npy")
        np.save(p_depth_npy, depth)
        np.save(p_mask_npy, mask)
        depth_vis = _depth_to_vis(depth, mask=mask)
        p_depth_vis_png = os.path.join(out_dir, "02_heic_depth_vis.png")
        cv2.imwrite(p_depth_vis_png, depth_vis)
        outputs.extend([p_depth_npy, p_mask_npy, p_depth_vis_png])
        if verbose:
            _print_stage_done((p_depth_npy, p_mask_npy, p_depth_vis_png))
    else:
        if verbose:
            _print_stage("Stage 3 | SMIRK 推理深度（depth / mask / 可视化）")
        if smirk_ckpt is None:
            raise ValueError("smirk_ckpt 未指定，且未提供 depth_override（HEIC 深度）")
        smirk_res = smirk_depth_from_cropped_bgr_224(crop_res.cropped_bgr_224, smirk_ckpt, device)
        depth = smirk_res["depth"]
        mask = smirk_res["mask"]
        p_depth_npy = os.path.join(out_dir, "02_depth.npy")
        p_mask_npy = os.path.join(out_dir, "02_depth_mask.npy")
        np.save(p_depth_npy, depth)
        np.save(p_mask_npy, mask)
        depth_vis = _depth_to_vis(depth, mask=mask)
        p_depth_vis_png = os.path.join(out_dir, "02_depth_vis.png")
        cv2.imwrite(p_depth_vis_png, depth_vis)
        p_smirk_mesh_png = os.path.join(out_dir, "02_smirk_mesh.png")
        cv2.imwrite(p_smirk_mesh_png, smirk_res["mesh_bgr"])
        p_smirk_mesh_overlay_png = os.path.join(out_dir, "02_smirk_mesh_overlay.png")
        cv2.imwrite(p_smirk_mesh_overlay_png, smirk_res["mesh_overlay_bgr"])
        p_smirk_raw_json = os.path.join(out_dir, "02_smirk_raw.json")
        _write_json(
            p_smirk_raw_json,
            {
                "smirk_outputs_keys": sorted(list(smirk_res["smirk_outputs"].keys())),
                "flame": {"vertices_shape": list(smirk_res["flame"]["vertices"].shape)},
            },
        )
        p_smirk_vertices_npy = os.path.join(out_dir, "02_smirk_vertices.npy")
        np.save(p_smirk_vertices_npy, smirk_res["flame"]["vertices"])
        outputs.extend([
            p_depth_npy,
            p_mask_npy,
            p_depth_vis_png,
            p_smirk_mesh_png,
            p_smirk_mesh_overlay_png,
            p_smirk_raw_json,
            p_smirk_vertices_npy,
        ])
        if verbose:
            _print_stage_done((
                p_depth_npy,
                p_mask_npy,
                p_depth_vis_png,
                p_smirk_mesh_png,
                p_smirk_mesh_overlay_png,
                p_smirk_raw_json,
                p_smirk_vertices_npy,
            ))

    if verbose:
        _print_stage("Stage 4 | DynaFace 推理关键点（landmarks / overlay，原分辨率）")
    dyna_res = dynaface_landmarks_and_overlay(crop_res.cropped_bgr_no_resize, device=None)
    p_dynaface_landmarks_json = os.path.join(out_dir, "03_dynaface_landmarks.json")
    _write_json(
        p_dynaface_landmarks_json,
        {"ok": dyna_res["ok"], "landmarks_xy": dyna_res["landmarks_xy"]},
    )
    outputs.append(p_dynaface_landmarks_json)
    stage4_outputs = [p_dynaface_landmarks_json]
    if dyna_res["overlay_bgr"] is not None:
        p_dynaface_overlay_png = os.path.join(out_dir, "03_dynaface_overlay.png")
        cv2.imwrite(p_dynaface_overlay_png, dyna_res["overlay_bgr"])
        outputs.append(p_dynaface_overlay_png)
        stage4_outputs.append(p_dynaface_overlay_png)
    if dyna_res["overlay_numbers_bgr"] is not None:
        p_dynaface_overlay_numbers_png = os.path.join(out_dir, "03_dynaface_overlay_numbers.png")
        cv2.imwrite(p_dynaface_overlay_numbers_png, dyna_res["overlay_numbers_bgr"])
        outputs.append(p_dynaface_overlay_numbers_png)
        stage4_outputs.append(p_dynaface_overlay_numbers_png)
    if verbose:
        _print_stage_done(tuple(stage4_outputs))

    if verbose:
        _print_stage("Stage 5 | FaceCheck 推理（分类结果）")
    from facecheck.inference.predictor import FaceCheckPredictor

    facecheck_ckpt_abs = _resolve_facecheck_ckpt(facecheck_ckpt, (os.getcwd(), _repo_root()))
    predictor = FaceCheckPredictor.load(facecheck_ckpt_abs, device=device)
    lm_vec = _landmark_vec_for_facecheck(predictor, crop_res.cropped_bgr_no_resize, dyna_res["landmarks_xy"])
    result = predictor.predict_bgr_depth(crop_res.cropped_bgr_224, depth, landmark_vec=lm_vec)
    p_facecheck_result_json = os.path.join(out_dir, "04_facecheck_result.json")
    p_landmark_vec_npy = os.path.join(out_dir, "04_landmark_vec.npy")
    _write_json(p_facecheck_result_json, {"prob_affected": result.prob_affected, "label": result.label})
    np.save(p_landmark_vec_npy, lm_vec.astype(np.float32))
    outputs.extend([p_facecheck_result_json, p_landmark_vec_npy])
    if verbose:
        _print_stage_done((p_facecheck_result_json, p_landmark_vec_npy))

    if verbose:
        _print_stage("Stage 6 | 导出汇总图（final.png）")
    overlay = dyna_res["overlay_bgr"]
    if overlay is None:
        overlay = crop_res.cropped_bgr_224.copy()
    overlay = cv2.resize(overlay, (224, 224), interpolation=cv2.INTER_AREA)
    overlay = _put_text(
        overlay,
        (f"FaceCheck: {result.label}", f"prob_affected={result.prob_affected:.4f}"),
    )

    final = _hstack((img_bgr, crop_res.cropped_bgr_224, depth_vis, overlay), height=512)
    p_final_png = os.path.join(out_dir, "final.png")
    cv2.imwrite(p_final_png, final)
    outputs.append(p_final_png)
    if verbose:
        _print_stage_done((p_final_png,))

    if verbose:
        _print_stage("Stage 7 | 打包输出（zip）")
    zip_path = os.path.abspath(out_dir.rstrip(os.sep) + ".zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in outputs:
            ap = os.path.abspath(p)
            if os.path.isfile(ap):
                zf.write(ap, arcname=os.path.relpath(ap, out_dir))
    outputs.append(zip_path)
    if verbose:
        _print_stage_done((zip_path,))

    return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--out_dir", default="output_infer", type=str)
    parser.add_argument("--smirk_ckpt", default=None, type=str,
                        help="SMIRK checkpoint（使用 HEIC 深度时可省略）")
    parser.add_argument("--facecheck_ckpt", default="outputs/facecheck/best.pt", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()

    out_dir = _ensure_dir(os.path.abspath(args.out_dir))

    _heic_exts = {".heic", ".heif"}
    ext = os.path.splitext(args.input)[1].lower()

    if ext in _heic_exts:
        print(f"检测到 HEIC 文件，自动提取 RGB / 深度 / 人像遮罩：{args.input}")
        img_bgr, heic_depth, heic_matte = load_heic_rgb_and_depth(args.input)
        if heic_depth is None:
            print("[警告] 未能从 HEIC 提取深度辅助图，将回退到 SMIRK 深度估计。")
            if args.smirk_ckpt is None:
                raise ValueError("HEIC 无深度且 --smirk_ckpt 未指定，无法继续。")
        run_pipeline(
            img_bgr=img_bgr,
            out_dir=out_dir,
            smirk_ckpt=args.smirk_ckpt,
            facecheck_ckpt=args.facecheck_ckpt,
            device=args.device,
            verbose=True,
            depth_override=heic_depth,
            face_matte_override=heic_matte,
        )
    else:
        img_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(args.input)
        if args.smirk_ckpt is None:
            raise ValueError("非 HEIC 输入时 --smirk_ckpt 为必填项。")
        run_pipeline(
            img_bgr=img_bgr,
            out_dir=out_dir,
            smirk_ckpt=args.smirk_ckpt,
            facecheck_ckpt=args.facecheck_ckpt,
            device=args.device,
            verbose=True,
        )


if __name__ == "__main__":
    main()
