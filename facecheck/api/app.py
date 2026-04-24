import os

from facecheck.api.routes import build_router
from facecheck.inference.predictor import FaceCheckPredictor


def create_app():
    from fastapi import FastAPI

    ckpt = os.environ.get("FACECHECK_CKPT", "")
    if not ckpt:
        raise ValueError("FACECHECK_CKPT is required")
    predictor = FaceCheckPredictor.load(ckpt)

    app = FastAPI(title="FaceCheck")
    app.include_router(build_router(predictor))
    return app


app = create_app()

