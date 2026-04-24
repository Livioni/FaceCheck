import json
from typing import Optional

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
