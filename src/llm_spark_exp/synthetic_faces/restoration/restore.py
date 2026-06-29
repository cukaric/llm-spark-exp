"""OSDFace one-step face restoration for synthetic face datasets."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from llm_spark_exp.paths import MODELS_DIR
from llm_spark_exp.synthetic_faces._common import iter_identity_dirs, list_identity_images

DEFAULT_OSDFACE_REPO_URL = "https://github.com/jkwang28/OSDFace.git"
HF_OSDFACE_WEIGHTS_REPO_ID = "alecccdd/OSDFace"
HF_SD21_BASE_REPO_ID = "stabilityai/stable-diffusion-2-1-base"
DEFAULT_OSDFACE_DIR = MODELS_DIR / "osdface"
DEFAULT_OSDFACE_REPO_DIR = DEFAULT_OSDFACE_DIR / "repo"
DEFAULT_OSDFACE_WEIGHTS_DIR = DEFAULT_OSDFACE_DIR / "weights"


@dataclass(frozen=True)
class FaceRestoreResult:
    source_dir: Path
    output_dir: Path
    identities: int
    source_images: int
    restored_images: int


@dataclass(frozen=True)
class FaceRestoreConfig:
    merge_lora: bool = True
    mixed_precision: str = "fp16"
    jpeg_quality: int = 95
    upscale: bool = False
    upscale_factor: int = 2


@dataclass(frozen=True)
class OSDFacePaths:
    repo_dir: Path = DEFAULT_OSDFACE_REPO_DIR
    weights_dir: Path = DEFAULT_OSDFACE_WEIGHTS_DIR
    sd_model_name_or_path: str = HF_SD21_BASE_REPO_ID
    python_executable: str = "python"

    @property
    def img_encoder_weight(self) -> Path:
        return self.weights_dir / "associate_2.ckpt"

    @property
    def ckpt_dir(self) -> Path:
        return self.weights_dir

    @property
    def lora_weights(self) -> Path:
        return self.weights_dir / "pytorch_lora_weights.safetensors"

    @property
    def embedding_change_weights(self) -> Path:
        return self.weights_dir / "embedding_change_weights.pth"


@dataclass(frozen=True)
class OSDFaceWeight:
    repo_id: str
    filename: str
    local_dir: Path


def default_osdface_weights(
    *,
    weights_dir: Path = DEFAULT_OSDFACE_WEIGHTS_DIR,
) -> tuple[OSDFaceWeight, ...]:
    return (
        OSDFaceWeight(
            repo_id=HF_OSDFACE_WEIGHTS_REPO_ID,
            filename="associate_2.ckpt",
            local_dir=weights_dir,
        ),
        OSDFaceWeight(
            repo_id=HF_OSDFACE_WEIGHTS_REPO_ID,
            filename="embedding_change_weights.pth",
            local_dir=weights_dir,
        ),
        OSDFaceWeight(
            repo_id=HF_OSDFACE_WEIGHTS_REPO_ID,
            filename="pytorch_lora_weights.safetensors",
            local_dir=weights_dir,
        ),
    )


def download_osdface_weights(
    *,
    weights_dir: Path = DEFAULT_OSDFACE_WEIGHTS_DIR,
) -> list[Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required for OSDFace weight downloads.") from error

    weights_dir.mkdir(parents=True, exist_ok=True)
    weights = default_osdface_weights(weights_dir=weights_dir)
    downloaded: list[Path] = []
    for weight in weights:
        local_path = weight.local_dir / weight.filename
        if local_path.exists():
            downloaded.append(local_path)
            continue
        path = Path(
            hf_hub_download(
                repo_id=weight.repo_id,
                filename=weight.filename,
                local_dir=str(weight.local_dir),
            )
        )
        downloaded.append(path)
    return downloaded


def clone_osdface_repo(
    *,
    repo_dir: Path = DEFAULT_OSDFACE_REPO_DIR,
    repo_url: str = DEFAULT_OSDFACE_REPO_URL,
) -> Path:
    if repo_dir.exists():
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["git", "clone", repo_url, str(repo_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Failed to clone OSDFace repo: {error.stderr}") from error
    except FileNotFoundError as error:
        raise RuntimeError("git is required to clone the OSDFace repository.") from error

    return repo_dir


def build_face_restore_command(
    *,
    source_dir: Path,
    output_dir: Path,
    repo_dir: Path,
    weights_dir: Path,
    sd_model_name_or_path: str = HF_SD21_BASE_REPO_ID,
    merge_lora: bool = True,
    mixed_precision: str = "fp16",
    jpeg_quality: int = 95,
    device: str = "cuda",
    upscale: bool = False,
    upscale_factor: int = 2,
    python_executable: str = "python",
) -> tuple[str, ...]:
    command = [
        python_executable,
        "-m",
        "llm_spark_exp.synthetic_faces.restoration.restore_worker",
        "--source-dir",
        str(source_dir),
        "--output-dir",
        str(output_dir),
        "--repo-dir",
        str(repo_dir),
        "--weights-dir",
        str(weights_dir),
        "--sd-model",
        sd_model_name_or_path,
        "--jpeg-quality",
        str(jpeg_quality),
        "--device",
        device,
    ]
    if merge_lora:
        command.append("--merge-lora")
    if mixed_precision:
        command.extend(["--mixed-precision", mixed_precision])
    if upscale:
        command.extend(["--upscale", "--upscale-factor", str(upscale_factor)])
    return tuple(command)


@dataclass(frozen=True)
class FaceRestoreRun:
    command: tuple[str, ...]
    source_dir: Path
    output_dir: Path
    stdout: str
    stderr: str
    returncode: int


class FaceRestoreRunner:
    def __init__(self, paths: OSDFacePaths | None = None) -> None:
        self.paths = paths or OSDFacePaths()

    def ensure_requirements(self) -> None:
        if not self.paths.repo_dir.exists():
            clone_osdface_repo(repo_dir=self.paths.repo_dir)
        if not self.paths.img_encoder_weight.exists():
            download_osdface_weights(weights_dir=self.paths.weights_dir)

    def build_command(
        self,
        *,
        source_dir: Path,
        output_dir: Path,
        config: FaceRestoreConfig | None = None,
        device: str = "cuda",
    ) -> tuple[str, ...]:
        config = config or FaceRestoreConfig()
        return build_face_restore_command(
            source_dir=source_dir,
            output_dir=output_dir,
            repo_dir=self.paths.repo_dir,
            weights_dir=self.paths.weights_dir,
            sd_model_name_or_path=self.paths.sd_model_name_or_path,
            merge_lora=config.merge_lora,
            mixed_precision=config.mixed_precision,
            jpeg_quality=config.jpeg_quality,
            device=device,
            upscale=config.upscale,
            upscale_factor=config.upscale_factor,
            python_executable=self.paths.python_executable,
        )

    def restore_dataset(
        self,
        *,
        source_dir: Path,
        output_dir: Path,
        config: FaceRestoreConfig | None = None,
        device: str = "cuda",
        env: dict[str, str] | None = None,
    ) -> FaceRestoreRun:
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

        self.ensure_requirements()
        config = config or FaceRestoreConfig()
        command = self.build_command(
            source_dir=source_dir,
            output_dir=output_dir,
            config=config,
            device=device,
        )
        completed = subprocess.run(
            command,
            env=dict(env) if env is not None else None,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "OSDFace restoration failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )

        return FaceRestoreRun(
            command=command,
            source_dir=source_dir,
            output_dir=output_dir,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )


def restore_identity_dataset(
    *,
    source_dir: Path,
    output_dir: Path,
    config: FaceRestoreConfig | None = None,
    paths: OSDFacePaths | None = None,
    device: str = "cuda",
) -> FaceRestoreResult:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    runner = FaceRestoreRunner(paths=paths)
    runner.restore_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        config=config,
        device=device,
    )

    identities = 0
    source_images = 0
    restored_images = 0
    if output_dir.exists():
        for identity_dir in iter_identity_dirs(output_dir):
            images = list_identity_images(identity_dir)
            if not images:
                continue
            identities += 1
            restored_images += len(images)

    for identity_dir in iter_identity_dirs(source_dir):
        images = list_identity_images(identity_dir)
        if images:
            source_images += len(images)

    return FaceRestoreResult(
        source_dir=source_dir,
        output_dir=output_dir,
        identities=identities,
        source_images=source_images,
        restored_images=restored_images,
    )
