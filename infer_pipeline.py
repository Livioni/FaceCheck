import argparse
import glob
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from skimage.transform import estimate_transform, warp


@dataclass(frozen=True)
class CropResult:
    tform_params: np.ndarray
    cropped_bgr_224: np.ndarray
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


def _import_smirk() -> Tuple[Any, Any, Any, Any, Any]:
    _smirk_dir()
    from src.smirk_encoder import SmirkEncoder  # type: ignore
    from src.FLAME.FLAME import FLAME  # type: ignore
    from src.renderer.util import batch_orth_proj  # type: ignore
    from utils.mediapipe_utils import run_mediapipe  # type: ignore

    from pytorch3d.structures import Meshes  # type: ignore
    from pytorch3d.renderer.mesh import rasterize_meshes  # type: ignore

    return SmirkEncoder, FLAME, batch_orth_proj, run_mediapipe, (Meshes, rasterize_meshes)


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
    SmirkEncoder, FLAME, batch_orth_proj, run_mediapipe, _ = _import_smirk()
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

    cropped = warp(img_bgr, tform.inverse, output_shape=(image_size, image_size), preserve_range=True).astype(np.uint8)
    cropped_kpt = (np.dot(tform.params, np.hstack([kpt_xy, np.ones([kpt_xy.shape[0], 1], dtype=np.float32)]).T).T)[
        :, :2
    ].astype(np.float32)

    cropped = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_AREA)

    return CropResult(
        tform_params=np.asarray(tform.params, dtype=np.float32),
        cropped_bgr_224=cropped,
        mediapipe_xy=kpt_xy,
        cropped_mediapipe_xy=cropped_kpt,
    )


@torch.no_grad()
def smirk_depth_from_cropped_bgr_224(
    cropped_bgr_224: np.ndarray,
    smirk_ckpt: str,
    device: str,
) -> Dict[str, Any]:
    SmirkEncoder, FLAME, batch_orth_proj, _, (Meshes, rasterize_meshes) = _import_smirk()

    smirk_dir = _smirk_dir()
    smirk_ckpt_abs = _resolve_existing_file(smirk_ckpt, (os.getcwd(), _repo_root(), smirk_dir))

    dev = torch.device(device)
    old = _pushd(smirk_dir)
    try:
        smirk_encoder = SmirkEncoder().to(dev).eval()
        ckpt = torch.load(smirk_ckpt_abs, map_location="cpu")
        encoder_state = {k.replace("smirk_encoder.", ""): v for k, v in ckpt.items() if "smirk_encoder" in k}
        smirk_encoder.load_state_dict(encoder_state, strict=True)

        flame = FLAME().to(dev).eval()

        rgb = cv2.cvtColor(cropped_bgr_224, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = x.to(dev)

        outputs = smirk_encoder(x)
        flame_out = flame.forward(outputs)
        vertices = flame_out["vertices"]
        faces = flame.faces_tensor.unsqueeze(0).expand(vertices.shape[0], -1, -1)

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
    model_path = models.download_models()
    models.init_models(model_path, dev)

    analyzer = facial.AnalyzeFace(measures=[measures.AnalyzeLandmarks()])
    ok = analyzer.load_image(img_bgr, crop=False)
    if not ok or analyzer.is_no_face():
        return {"ok": False, "landmarks_xy": None, "landmarks_dict": None, "overlay_bgr": None}

    lm_dict = analyzer.analyze()
    if lm_dict is None:
        return {"ok": False, "landmarks_xy": None, "landmarks_dict": None, "overlay_bgr": None}

    n = int(getattr(measures.AnalyzeLandmarks, "NUM_LANDMARKS", 97))
    xy = np.zeros((n, 2), dtype=np.float32)
    for i in range(1, n + 1):
        xy[i - 1, 0] = float(lm_dict.get(f"landmark-{i}-x", 0.0))
        xy[i - 1, 1] = float(lm_dict.get(f"landmark-{i}-y", 0.0))

    overlay = analyzer.render_img.copy()
    return {"ok": True, "landmarks_xy": xy, "landmarks_dict": lm_dict, "overlay_bgr": overlay}


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--out_dir", default="output_infer", type=str)
    parser.add_argument("--smirk_ckpt", required=True, type=str)
    parser.add_argument("--facecheck_ckpt", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()

    out_dir = _ensure_dir(os.path.abspath(args.out_dir))
    _print_stage("Stage 0 | 初始化")
    _print_stage_done((out_dir,))

    _print_stage("Stage 1 | 读取输入并保存原图")
    img_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(args.input)

    p_input_png = os.path.join(out_dir, "00_input.png")
    cv2.imwrite(p_input_png, img_bgr)
    _print_stage_done((p_input_png,))

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
    _print_stage_done((p_cropped_png, p_crop_json))

    _print_stage("Stage 3 | SMIRK 推理深度（depth / mask / 可视化）")
    smirk_res = smirk_depth_from_cropped_bgr_224(crop_res.cropped_bgr_224, args.smirk_ckpt, args.device)
    depth = smirk_res["depth"]
    mask = smirk_res["mask"]
    p_depth_npy = os.path.join(out_dir, "02_depth.npy")
    p_mask_npy = os.path.join(out_dir, "02_depth_mask.npy")
    np.save(p_depth_npy, depth)
    np.save(p_mask_npy, mask)
    depth_vis = _depth_to_vis(depth, mask=mask)
    p_depth_vis_png = os.path.join(out_dir, "02_depth_vis.png")
    cv2.imwrite(p_depth_vis_png, depth_vis)
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
    _print_stage_done((p_depth_npy, p_mask_npy, p_depth_vis_png, p_smirk_raw_json, p_smirk_vertices_npy))

    _print_stage("Stage 4 | DynaFace 推理关键点（landmarks / overlay）")
    dyna_res = dynaface_landmarks_and_overlay(crop_res.cropped_bgr_224, device=None)
    p_dynaface_landmarks_json = os.path.join(out_dir, "03_dynaface_landmarks.json")
    _write_json(
        p_dynaface_landmarks_json,
        {"ok": dyna_res["ok"], "landmarks_xy": dyna_res["landmarks_xy"]},
    )
    stage4_outputs = [p_dynaface_landmarks_json]
    if dyna_res["overlay_bgr"] is not None:
        p_dynaface_overlay_png = os.path.join(out_dir, "03_dynaface_overlay.png")
        cv2.imwrite(p_dynaface_overlay_png, dyna_res["overlay_bgr"])
        stage4_outputs.append(p_dynaface_overlay_png)
    _print_stage_done(tuple(stage4_outputs))

    _print_stage("Stage 5 | FaceCheck 推理（分类结果）")
    from facecheck.inference.predictor import FaceCheckPredictor

    facecheck_ckpt_abs = _resolve_facecheck_ckpt(args.facecheck_ckpt, (os.getcwd(), _repo_root()))
    predictor = FaceCheckPredictor.load(facecheck_ckpt_abs, device=args.device)
    lm_vec = _landmark_vec_for_facecheck(predictor, crop_res.cropped_bgr_224, dyna_res["landmarks_xy"])
    result = predictor.predict_bgr_depth(crop_res.cropped_bgr_224, depth, landmark_vec=lm_vec)
    p_facecheck_result_json = os.path.join(out_dir, "04_facecheck_result.json")
    p_landmark_vec_npy = os.path.join(out_dir, "04_landmark_vec.npy")
    _write_json(p_facecheck_result_json, {"prob_affected": result.prob_affected, "label": result.label})
    np.save(p_landmark_vec_npy, lm_vec.astype(np.float32))
    _print_stage_done((p_facecheck_result_json, p_landmark_vec_npy))

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
    _print_stage_done((p_final_png,))


if __name__ == "__main__":
    main()
