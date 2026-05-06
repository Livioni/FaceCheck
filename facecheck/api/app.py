import os

from facecheck.api.routes import build_heic_router, build_pipeline_router, build_quicktest_router, build_router
from facecheck.inference.predictor import FaceCheckPredictor


def create_app():
    from fastapi import FastAPI

    ckpt_env = os.environ.get("FACECHECK_CKPT", "").strip()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    ckpt_candidates: list[str] = []
    if ckpt_env:
        ckpt_candidates.append(os.path.expandvars(os.path.expanduser(ckpt_env)))
        if not os.path.isabs(ckpt_candidates[0]):
            ckpt_candidates.append(os.path.join(project_root, ckpt_candidates[0]))
    ckpt_candidates.append(os.path.join(project_root, "outputs", "facecheck", "best.pt"))

    ckpt = next((p for p in ckpt_candidates if os.path.exists(p)), "")
    if not ckpt:
        raise ValueError(
            "FaceCheck checkpoint not found. "
            "Set FACECHECK_CKPT to a valid .pt path. "
            f"Tried: {ckpt_candidates}"
        )

    predictor = FaceCheckPredictor.load(ckpt)

    app = FastAPI(title="FaceCheck")
    app.include_router(build_router(predictor))

    smirk_env = os.environ.get("SMIRK_CKPT", "").strip()
    smirk_candidates: list[str] = []
    if smirk_env:
        smirk_candidates.append(os.path.expandvars(os.path.expanduser(smirk_env)))
        if not os.path.isabs(smirk_candidates[0]):
            smirk_candidates.append(os.path.join(project_root, smirk_candidates[0]))
    smirk_candidates.append(os.path.join(project_root, "smirk", "pretrained_models", "SMIRK_em1.pt"))

    smirk_ckpt = next((p for p in smirk_candidates if os.path.exists(p)), "")
    output_root = os.environ.get("OUTPUT_ROOT", "/tmp/facecheck_outputs").strip() or "/tmp/facecheck_outputs"
    device = os.environ.get("DEVICE", "cuda").strip() or "cuda"

    app.include_router(build_quicktest_router(output_root=output_root))

    # /heic — uses embedded HEIC depth, no SMIRK checkpoint required
    app.include_router(
        build_heic_router(facecheck_ckpt=ckpt, device=device, output_root=output_root)
    )

    # /pipeline — uses SMIRK for depth estimation, requires SMIRK checkpoint
    if smirk_ckpt:
        app.include_router(
            build_pipeline_router(
                smirk_ckpt=smirk_ckpt,
                facecheck_ckpt=ckpt,
                device=device,
                output_root=output_root,
            )
        )

    return app


app = create_app()
