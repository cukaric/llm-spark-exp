"""Shared fixtures and GPU gating for the synthetic-face worker tests.

GPU integration tests are marked ``@pytest.mark.gpu`` and are skipped unless
``--run-gpu`` is passed, so the default ``pytest`` run stays fast and offline.
Each GPU fixture additionally skips when its CUDA device or local model
weights/repos are missing, so the tests degrade cleanly on an unprovisioned box.
"""

from __future__ import annotations

import pytest

from llm_spark_exp.synthetic_faces.generation.vec2face_plus import Vec2FacePlusPaths
from llm_spark_exp.synthetic_faces.restoration.restore import OSDFacePaths


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run @pytest.mark.gpu tests (need a CUDA device and local model weights/repos)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-gpu"):
        return
    skip_gpu = pytest.mark.skip(reason="needs --run-gpu (CUDA device + model weights/repos)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def gpu_device():
    """Return a CUDA device, skipping the test when none is available."""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def vec2face_paths() -> Vec2FacePlusPaths:
    """Vec2Face+ paths, skipping when the repo or face-recognition weights are absent."""

    paths = Vec2FacePlusPaths()
    required = [
        paths.repo_dir,
        paths.weights_dir / "weights" / "arcface-r100-glint360k.pth",
        paths.weights_dir / "weights" / "magface-r100-glint360k.pth",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip(f"Vec2Face+ assets missing: {', '.join(str(p) for p in missing)}")
    return paths


@pytest.fixture(scope="session")
def osdface_paths() -> OSDFacePaths:
    """OSDFace paths, skipping when repo/weights are missing or SD2.1 base is not cached."""

    paths = OSDFacePaths()
    required = [
        paths.repo_dir,
        paths.img_encoder_weight,
        paths.embedding_change_weights,
        paths.lora_weights,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip(f"OSDFace assets missing: {', '.join(str(p) for p in missing)}")

    # Avoid triggering a multi-GB Stable Diffusion download from within a test.
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(paths.sd_model_name_or_path, "model_index.json")
    if not isinstance(cached, str):
        pytest.skip("stable-diffusion-2-1-base not in local HF cache (would download ~5GB)")
    return paths
