import json
import os
import shutil
import uuid
import zipfile
from typing import Any, Optional

import numpy as np

from facecheck.api.models import HealthResponse, PredictResponse


def build_router(predictor):
    from fastapi import APIRouter, File, Form, UploadFile

    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @router.post("/predict", response_model=PredictResponse)
    async def predict(
        image: UploadFile = File(...),
        depth: UploadFile = File(...),
        landmark: Optional[str] = Form(None),
    ) -> PredictResponse:
        import cv2

        img_bytes = await image.read()
        depth_bytes = await depth.read()

        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Invalid image")

        try:
            from io import BytesIO

            depth_np = np.load(BytesIO(depth_bytes))
        except Exception as e:
            raise ValueError("Invalid depth npy") from e

        lm_vec = None
        if landmark is not None:
            v = np.asarray(json.loads(landmark), dtype=np.float32).reshape(-1)
            lm_vec = v

        out = predictor.predict_bgr_depth(img_bgr, depth_np, landmark_vec=lm_vec)
        return PredictResponse(prob_affected=out.prob_affected, label=out.label)

    return router


def build_pipeline_router(
    *,
    smirk_ckpt: str,
    facecheck_ckpt: str,
    device: str,
    output_root: str,
) -> Any:
    from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse

    router = APIRouter()

    def _cleanup(paths: tuple[str, ...]) -> None:
        for p in paths:
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass

    @router.post("/pipeline")
    async def pipeline(background_tasks: BackgroundTasks, image: UploadFile = File(...)) -> FileResponse:
        import cv2
        import infer_pipeline

        img_bytes = await image.read()
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        job_id = uuid.uuid4().hex
        job_dir = os.path.join(os.path.abspath(output_root), job_id)
        zip_path = f"{job_dir}.zip"

        try:
            os.makedirs(job_dir, exist_ok=True)
            infer_pipeline.run_pipeline(
                img_bgr=img_bgr,
                out_dir=job_dir,
                smirk_ckpt=smirk_ckpt,
                facecheck_ckpt=facecheck_ckpt,
                device=device,
                verbose=False,
            )

            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(job_dir):
                    for fn in files:
                        full = os.path.join(root, fn)
                        rel = os.path.relpath(full, job_dir)
                        zf.write(full, arcname=rel)
        except HTTPException:
            raise
        except RuntimeError as e:
            _cleanup((job_dir, zip_path))
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            _cleanup((job_dir, zip_path))
            raise HTTPException(status_code=500, detail=str(e)) from e

        background_tasks.add_task(_cleanup, (job_dir, zip_path))
        return FileResponse(zip_path, media_type="application/zip", filename=f"{job_id}.zip")

    return router


def build_heic_router(
    *,
    facecheck_ckpt: str,
    device: str,
    output_root: str,
) -> Any:
    """Router for /heic — accepts an iPhone HEIC file, extracts embedded depth + portrait matte,
    applies face-region masking, normalises, and runs the full FaceCheck pipeline."""
    import tempfile

    from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse

    router = APIRouter()

    def _cleanup(paths: tuple[str, ...]) -> None:
        for p in paths:
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass

    @router.post("/heic")
    async def heic_pipeline(
        background_tasks: BackgroundTasks,
        image: UploadFile = File(..., description="iPhone HEIC file with embedded depth"),
    ) -> FileResponse:
        import infer_pipeline

        heic_bytes = await image.read()

        # Write to a temp file because pillow_heif needs a file path
        suffix = os.path.splitext(image.filename or "upload.heic")[1].lower() or ".heic"
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(heic_bytes)

            try:
                img_bgr, heic_depth, heic_matte = infer_pipeline.load_heic_rgb_and_depth(tmp_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read HEIC: {e}") from e

            if heic_depth is None:
                raise HTTPException(
                    status_code=422,
                    detail="No depth auxiliary image found in HEIC file. "
                           "Use /pipeline (SMIRK-based depth) for plain RGB photos.",
                )
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        job_id = uuid.uuid4().hex
        job_dir = os.path.join(os.path.abspath(output_root), job_id)
        zip_path = f"{job_dir}.zip"

        try:
            os.makedirs(job_dir, exist_ok=True)
            infer_pipeline.run_pipeline(
                img_bgr=img_bgr,
                out_dir=job_dir,
                smirk_ckpt=None,
                facecheck_ckpt=facecheck_ckpt,
                device=device,
                verbose=False,
                depth_override=heic_depth,
                face_matte_override=heic_matte,
            )

            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(job_dir):
                    for fn in files:
                        full = os.path.join(root, fn)
                        zf.write(full, arcname=os.path.relpath(full, job_dir))
        except HTTPException:
            raise
        except RuntimeError as e:
            _cleanup((job_dir, zip_path))
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            _cleanup((job_dir, zip_path))
            raise HTTPException(status_code=500, detail=str(e)) from e

        background_tasks.add_task(_cleanup, (job_dir, zip_path))
        return FileResponse(zip_path, media_type="application/zip", filename=f"{job_id}.zip")

    return router


def build_quicktest_router(*, output_root: str) -> Any:
    from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse

    router = APIRouter()

    def _cleanup(paths: tuple[str, ...]) -> None:
        for p in paths:
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass

    @router.post("/quicktest")
    async def quicktest(background_tasks: BackgroundTasks, image: UploadFile = File(...)) -> FileResponse:
        import cv2
        import infer_pipeline

        img_bytes = await image.read()
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        job_id = uuid.uuid4().hex
        job_dir = os.path.join(os.path.abspath(output_root), job_id)
        out_path = os.path.join(job_dir, "01_dynaface_quicktest.jpg")
        zip_path = f"{job_dir}.zip"

        try:
            os.makedirs(job_dir, exist_ok=True)
            crop_res = infer_pipeline.crop_like_smirk_demo(img_bgr, image_size=224, scale=1.4)
            res = infer_pipeline.dynaface_quicktest_save(
                crop_res.cropped_bgr_no_resize, out_path, device=None
            )
            if not res.get("ok") or not res.get("save_path") or not os.path.isfile(out_path):
                _cleanup((job_dir, zip_path))
                raise HTTPException(status_code=400, detail="dynaface quicktest failed")

            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(out_path, arcname=os.path.basename(out_path))
        except HTTPException:
            raise
        except RuntimeError as e:
            _cleanup((job_dir, zip_path))
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            _cleanup((job_dir, zip_path))
            raise HTTPException(status_code=500, detail=str(e)) from e

        background_tasks.add_task(_cleanup, (job_dir, zip_path))
        return FileResponse(zip_path, media_type="application/zip", filename=f"{job_id}.zip")

    return router
