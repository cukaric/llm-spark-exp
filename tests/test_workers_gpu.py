"""GPU integration tests for the synthetic-face workers.

All tests here are marked ``gpu`` (run with ``pytest --run-gpu``) and rely on the
``gpu_device`` / ``vec2face_paths`` / ``osdface_paths`` fixtures, which skip when a
CUDA device or the local model weights/repos are unavailable. Heavy third-party
imports happen inside the tests so the module always collects.
"""

import sys

import pytest


@pytest.mark.gpu
def test_vec2face_fr_model_runs_forward(vec2face_paths, gpu_device) -> None:
    """The Vec2Face+ face-recognition model loads and produces a 512-d embedding."""

    import torch

    from llm_spark_exp.synthetic_faces.generation.center_worker import create_fr_model

    repo_dir = vec2face_paths.repo_dir.resolve()
    sys.path.insert(0, str(repo_dir))
    try:
        weights = vec2face_paths.weights_dir / "weights" / "arcface-r100-glint360k.pth"
        model = create_fr_model(weights).to(gpu_device)
        with torch.no_grad():
            embedding = model(torch.randn(1, 3, 112, 112, device=gpu_device))
    finally:
        if str(repo_dir) in sys.path:
            sys.path.remove(str(repo_dir))

    assert embedding.shape == (1, 512)
    assert torch.isfinite(embedding).all()


@pytest.mark.gpu
@pytest.mark.slow
def test_osdface_restore_image_shape_and_range(osdface_paths, gpu_device) -> None:
    """OSDFace one-step restoration returns a 512x512 image in [0, 1]."""

    import torch

    from llm_spark_exp.synthetic_faces.restoration.restore_worker import (
        load_osdface_model,
        restore_image,
    )

    model = load_osdface_model(
        repo_dir=osdface_paths.repo_dir,
        weights_dir=osdface_paths.weights_dir,
        sd_model=osdface_paths.sd_model_name_or_path,
        merge_lora=True,
        mixed_precision="fp16",
        device="cuda",
    )
    output = restore_image(model, torch.rand(1, 3, 64, 64) * 2 - 1)

    assert output.shape == (1, 3, 512, 512)
    assert torch.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0


@pytest.mark.gpu
def test_swinir_upscale_doubles_resolution(gpu_device) -> None:
    """SwinIR upscaling doubles spatial dims (requires the optional basicsr dep)."""

    pytest.importorskip("basicsr")
    import torch

    from llm_spark_exp.synthetic_faces.restoration.restore_worker import (
        load_swinir_model,
        upscale_image,
    )

    model = load_swinir_model(device="cuda", upscale_factor=2)
    output = upscale_image(model, torch.rand(1, 3, 16, 16), upscale_factor=2)

    assert output.shape[-2:] == (32, 32)
