"""Orchestration utilities for Vec2Face+ synthetic face generation."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_spark_exp.paths import MODELS_DIR

DEFAULT_VEC2FACE_PLUS_REPO_URL = "https://github.com/HaiyuWu/Vec2Face_plus"
HF_VEC2FACE_REPO_ID = "BooBooWu/Vec2Face"
HF_VEC2FACE_DATASET_REPO_ID = "BooBooWu/Vec2Face"
HF_VEC2FACE_PLUS_REPO_ID = "BooBooWu/Vec2Face_plus"
HF_VEC2FACE_PLUS_DATASET_REPO_ID = "BooBooWu/Vec2Face_plus"
HF_VEC2FACE_CENTER_FEATURE_EXAMPLE = "center_feature_examples.npy"
DEFAULT_VEC2FACE_PLUS_DIR = MODELS_DIR / "vec2face_plus" / "repo"
DEFAULT_VEC2FACE_PLUS_WEIGHTS_DIR = MODELS_DIR / "vec2face_plus"
DEFAULT_CENTER_FEATURES_PATH = (
    DEFAULT_VEC2FACE_PLUS_WEIGHTS_DIR / HF_VEC2FACE_CENTER_FEATURE_EXAMPLE
)
VFACE_DATASETS = {
    "10k": "VFaces/vface10k.lmdb",
    "20k": "VFaces/vface20k.lmdb",
    "100k": "VFaces/vface100k.lmdb",
}
HSFACE_DATASETS = {
    "10k": "HSFaces/hsface10k.lmdb",
    "20k": "HSFaces/hsface20k.lmdb",
}


@dataclass(frozen=True)
class Vec2FacePlusWeight:
    """A Hugging Face file needed by the upstream Vec2Face+ scripts."""

    repo_id: str
    filename: str
    local_dir: Path


def default_vec2face_plus_weights(
    *,
    weights_dir: Path = DEFAULT_VEC2FACE_PLUS_WEIGHTS_DIR,
) -> tuple[Vec2FacePlusWeight, ...]:
    """Return the model weights expected by the Vec2Face+ generation scripts."""

    return (
        Vec2FacePlusWeight(
            repo_id=HF_VEC2FACE_REPO_ID,
            filename="weights/6DRepNet_300W_LP_AFLW2000.pth",
            local_dir=weights_dir,
        ),
        Vec2FacePlusWeight(
            repo_id=HF_VEC2FACE_REPO_ID,
            filename="weights/arcface-r100-glint360k.pth",
            local_dir=weights_dir,
        ),
        Vec2FacePlusWeight(
            repo_id=HF_VEC2FACE_REPO_ID,
            filename="weights/magface-r100-glint360k.pth",
            local_dir=weights_dir,
        ),
        Vec2FacePlusWeight(
            repo_id=HF_VEC2FACE_PLUS_REPO_ID,
            filename="vec2face_plus/main_model.pth",
            local_dir=weights_dir,
        ),
        Vec2FacePlusWeight(
            repo_id=HF_VEC2FACE_PLUS_REPO_ID,
            filename="vec2face_plus/pose_model.pth",
            local_dir=weights_dir,
        ),
    )


@dataclass(frozen=True)
class Vec2FacePlusPaths:
    """Filesystem locations used to run an external Vec2Face+ checkout."""

    repo_dir: Path = DEFAULT_VEC2FACE_PLUS_DIR
    weights_dir: Path = DEFAULT_VEC2FACE_PLUS_WEIGHTS_DIR
    python_executable: str = "python"

    @property
    def pose_control_script(self) -> Path:
        return self.repo_dir / "pose_control.py"

    @property
    def pose_model_weights(self) -> Path:
        return self.weights_dir / "vec2face_plus" / "pose_model.pth"

    @property
    def main_model_weights(self) -> Path:
        return self.weights_dir / "vec2face_plus" / "main_model.pth"

    @property
    def generated_images_root(self) -> Path:
        return self.repo_dir / "generated_images_ref"

    @property
    def generated_center_images_root(self) -> Path:
        return self.repo_dir / "generated_images"

    def generated_images_dir(self, name: str) -> Path:
        return self.generated_images_root / name

    def generated_center_images_dir(self, name: str) -> Path:
        return self.generated_center_images_root / name


@dataclass(frozen=True)
class Vec2FaceRun:
    """A completed Vec2Face+ command invocation."""

    command: tuple[str, ...]
    repo_dir: Path
    output_dir: Path
    stdout: str
    stderr: str
    returncode: int


def build_pose_generation_command(
    *,
    image_file: Path,
    pose_file: Path,
    model_weights: Path,
    name: str,
    batch_size: int = 64,
    examples: int = 1,
    use_lora: bool = True,
    lora_r: int = 8,
    python_executable: str = "python",
) -> tuple[str, ...]:
    """Build the upstream Vec2Face+ pose-control command."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if examples < 1:
        raise ValueError("examples must be at least 1.")
    if lora_r < 1:
        raise ValueError("lora_r must be at least 1.")
    if not name.strip():
        raise ValueError("name must not be blank.")

    command = [
        python_executable,
        "pose_control.py",
        "--batch_size",
        str(batch_size),
        "--image_file",
        str(image_file),
        "--example",
        str(examples),
        "--name",
        name,
        "--model_weights",
        str(model_weights),
        "--pose_file",
        str(pose_file),
    ]
    if use_lora:
        command.extend(["--use_lora", "--lora_r", str(lora_r)])

    return tuple(command)


def build_center_feature_generation_command(
    *,
    repo_dir: Path,
    center_feature: Path,
    model_weights: Path,
    name: str,
    batch_size: int = 64,
    examples: int = 1,
    start_end: str | None = None,
    python_executable: str = "python",
    device: str = "cuda",
    skip_feature_refinement: bool = False,
    variation_sigmas: str = "0.08,0.12,0.18",
    variation_weights: str = "0.5,0.35,0.15",
) -> tuple[str, ...]:
    """Build the command for generating identities from center-feature vectors."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if examples < 1:
        raise ValueError("examples must be at least 1.")
    if not name.strip():
        raise ValueError("name must not be blank.")

    command = [
        python_executable,
        "-m",
        "llm_spark_exp.synthetic_faces.generation.center_worker",
        "--repo-dir",
        str(repo_dir),
        "--center-feature",
        str(center_feature),
        "--model-weights",
        str(model_weights),
        "--batch-size",
        str(batch_size),
        "--examples",
        str(examples),
        "--name",
        name,
        "--device",
        device,
        "--variation-sigmas",
        variation_sigmas,
        "--variation-weights",
        variation_weights,
    ]
    if start_end is not None:
        command.extend(["--start-end", start_end])
    if skip_feature_refinement:
        command.append("--skip-feature-refinement")

    return tuple(command)


class Vec2FacePlusRunner:
    """Run synthetic face generation through a local Vec2Face+ checkout."""

    def __init__(self, paths: Vec2FacePlusPaths | None = None) -> None:
        self.paths = paths or Vec2FacePlusPaths()

    def build_pose_command(
        self,
        *,
        image_file: Path,
        pose_file: Path,
        name: str,
        batch_size: int = 64,
        examples: int = 1,
        use_lora: bool = True,
        lora_r: int = 8,
    ) -> tuple[str, ...]:
        return build_pose_generation_command(
            image_file=image_file,
            pose_file=pose_file,
            model_weights=self.paths.pose_model_weights,
            name=name,
            batch_size=batch_size,
            examples=examples,
            use_lora=use_lora,
            lora_r=lora_r,
            python_executable=self.paths.python_executable,
        )

    def generate_with_pose(
        self,
        *,
        image_file: Path,
        pose_file: Path,
        name: str,
        batch_size: int = 64,
        examples: int = 1,
        use_lora: bool = True,
        lora_r: int = 8,
        env: Mapping[str, str] | None = None,
    ) -> Vec2FaceRun:
        """Generate synthetic faces using Vec2Face+'s pose-control script."""

        self._validate_runtime_paths(
            image_file=image_file,
            pose_file=pose_file,
            model_weights=self.paths.pose_model_weights,
        )
        command = self.build_pose_command(
            image_file=image_file,
            pose_file=pose_file,
            name=name,
            batch_size=batch_size,
            examples=examples,
            use_lora=use_lora,
            lora_r=lora_r,
        )
        completed = subprocess.run(
            command,
            cwd=self.paths.repo_dir,
            env=dict(env) if env is not None else None,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Vec2Face+ generation failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )

        return Vec2FaceRun(
            command=command,
            repo_dir=self.paths.repo_dir,
            output_dir=self.paths.generated_images_dir(name),
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )

    def build_center_feature_command(
        self,
        *,
        center_feature: Path,
        name: str,
        batch_size: int = 64,
        examples: int = 1,
        start_end: str | None = None,
        device: str = "cuda",
        skip_feature_refinement: bool = False,
        variation_sigmas: str = "0.08,0.12,0.18",
        variation_weights: str = "0.5,0.35,0.15",
    ) -> tuple[str, ...]:
        return build_center_feature_generation_command(
            repo_dir=self.paths.repo_dir,
            center_feature=center_feature,
            model_weights=self.paths.main_model_weights,
            name=name,
            batch_size=batch_size,
            examples=examples,
            start_end=start_end,
            python_executable=self.paths.python_executable,
            device=device,
            skip_feature_refinement=skip_feature_refinement,
            variation_sigmas=variation_sigmas,
            variation_weights=variation_weights,
        )

    def generate_from_center_features(
        self,
        *,
        center_feature: Path,
        name: str,
        batch_size: int = 64,
        examples: int = 1,
        start_end: str | None = None,
        device: str = "cuda",
        skip_feature_refinement: bool = False,
        variation_sigmas: str = "0.08,0.12,0.18",
        variation_weights: str = "0.5,0.35,0.15",
        env: Mapping[str, str] | None = None,
    ) -> Vec2FaceRun:
        """Generate new identities from center-feature vectors."""

        self._validate_center_feature_paths(center_feature=center_feature)
        command = self.build_center_feature_command(
            center_feature=center_feature,
            name=name,
            batch_size=batch_size,
            examples=examples,
            start_end=start_end,
            device=device,
            skip_feature_refinement=skip_feature_refinement,
            variation_sigmas=variation_sigmas,
            variation_weights=variation_weights,
        )
        completed = subprocess.run(
            command,
            cwd=self.paths.repo_dir,
            env=dict(env) if env is not None else None,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Vec2Face+ center-feature generation failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )

        return Vec2FaceRun(
            command=command,
            repo_dir=self.paths.repo_dir,
            output_dir=self.paths.generated_center_images_dir(name),
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )

    def _validate_runtime_paths(
        self,
        *,
        image_file: Path,
        pose_file: Path,
        model_weights: Path,
    ) -> None:
        missing = [
            path
            for path in (
                self.paths.repo_dir,
                self.paths.pose_control_script,
                image_file,
                pose_file,
                model_weights,
            )
            if not path.exists()
        ]
        if missing:
            formatted_paths = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing Vec2Face+ runtime paths: {formatted_paths}")

    def _validate_center_feature_paths(self, *, center_feature: Path) -> None:
        missing = [
            path
            for path in (
                self.paths.repo_dir,
                center_feature,
                self.paths.main_model_weights,
                self.paths.weights_dir / "weights" / "arcface-r100-glint360k.pth",
                self.paths.weights_dir / "weights" / "magface-r100-glint360k.pth",
            )
            if not path.exists()
        ]
        if missing:
            formatted_paths = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing Vec2Face+ center-feature paths: {formatted_paths}")


def build_huggingface_download_commands(
    weights: Sequence[Vec2FacePlusWeight],
) -> tuple[str, ...]:
    """Return shell-friendly Python snippets for downloading Vec2Face+ weights."""

    return tuple(
        "from huggingface_hub import hf_hub_download; "
        f"hf_hub_download(repo_id={weight.repo_id!r}, "
        f"filename={weight.filename!r}, local_dir={str(weight.local_dir)!r})"
        for weight in weights
    )


def download_huggingface_file(
    *,
    repo_id: str,
    filename: str,
    local_dir: Path,
    repo_type: str | None = None,
) -> Path:
    """Download one Hugging Face file and return the local path."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required for downloads.") from error

    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            repo_type=repo_type,
        )
    )


def download_center_feature_examples(
    *,
    local_dir: Path = DEFAULT_VEC2FACE_PLUS_WEIGHTS_DIR,
) -> Path:
    """Download the upstream example center-feature vectors."""

    return download_huggingface_file(
        repo_id=HF_VEC2FACE_REPO_ID,
        filename=HF_VEC2FACE_CENTER_FEATURE_EXAMPLE,
        local_dir=local_dir,
    )


def download_vface_lmdb(
    *,
    dataset_size: str,
    local_dir: Path = DEFAULT_VEC2FACE_PLUS_WEIGHTS_DIR,
) -> Path:
    """Download one released Vec2Face+ VFace LMDB dataset."""

    try:
        filename = VFACE_DATASETS[dataset_size]
    except KeyError as error:
        choices = ", ".join(sorted(VFACE_DATASETS))
        raise ValueError(f"dataset_size must be one of: {choices}") from error

    return download_huggingface_file(
        repo_id=HF_VEC2FACE_PLUS_DATASET_REPO_ID,
        filename=filename,
        local_dir=local_dir,
        repo_type="dataset",
    )


def download_hsface_lmdb(
    *,
    dataset_size: str,
    local_dir: Path = DEFAULT_VEC2FACE_PLUS_WEIGHTS_DIR,
) -> Path:
    """Download one released Vec2Face HSFace LMDB dataset."""

    try:
        filename = HSFACE_DATASETS[dataset_size]
    except KeyError as error:
        choices = ", ".join(sorted(HSFACE_DATASETS))
        raise ValueError(f"dataset_size must be one of: {choices}") from error

    return download_huggingface_file(
        repo_id=HF_VEC2FACE_DATASET_REPO_ID,
        filename=filename,
        local_dir=local_dir,
        repo_type="dataset",
    )


@dataclass(frozen=True)
class LmdbImageExport:
    """Summary of images exported from an LMDB dataset."""

    lmdb_path: Path
    output_dir: Path
    images: int


def export_lmdb_images(
    *,
    lmdb_path: Path,
    output_dir: Path,
    limit: int | None = None,
) -> LmdbImageExport:
    """Export encoded images from a Vec2Face/Vec2Face+ LMDB dataset."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1.")
    try:
        import lmdb
        import msgpack
    except ImportError as error:
        raise RuntimeError("lmdb and msgpack are required to export LMDB images.") from error

    if not lmdb_path.exists():
        raise FileNotFoundError(f"LMDB path does not exist: {lmdb_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=lmdb_path.is_dir(),
    )
    exported = 0
    per_label_counts: dict[str, int] = {}
    with env.begin(write=False) as txn:
        keys = _lmdb_record_keys(txn)
        for key in keys:
            value = txn.get(key)
            if value is None:
                continue
            image_bytes, label = _unpack_lmdb_image(value, msgpack=msgpack)
            label_name = "unknown" if label is None else f"{int(label):06d}"
            label_dir = output_dir / label_name
            label_dir.mkdir(parents=True, exist_ok=True)
            image_index = per_label_counts.get(label_name, 0)
            (label_dir / f"{image_index:03d}.jpg").write_bytes(image_bytes)
            per_label_counts[label_name] = image_index + 1
            exported += 1
            if limit is not None and exported >= limit:
                break
    env.close()

    return LmdbImageExport(lmdb_path=lmdb_path, output_dir=output_dir, images=exported)


def _lmdb_record_keys(txn: Any) -> Iterable[bytes]:
    packed_keys = txn.get(b"__keys__")
    if packed_keys is not None:
        try:
            import msgpack

            keys = msgpack.unpackb(packed_keys, raw=False)
            return [key if isinstance(key, bytes) else str(key).encode("ascii") for key in keys]
        except Exception:
            pass

    packed_length = txn.get(b"__len__")
    if packed_length is not None:
        try:
            import msgpack

            length = msgpack.unpackb(packed_length, raw=False)
            return [str(index).encode("ascii") for index in range(int(length))]
        except Exception:
            pass

    return [key for key, _ in txn.cursor() if not key.startswith(b"__")]


def _unpack_lmdb_image(value: bytes, *, msgpack: Any) -> tuple[bytes, int | None]:
    record = msgpack.unpackb(value, raw=False)
    if isinstance(record, Mapping):
        image = record.get("image") or record.get("img") or record.get("data")
        label = record.get("label")
    elif isinstance(record, Sequence):
        image = record[0]
        label = record[2] if len(record) >= 3 else None
    else:
        raise ValueError(f"Unsupported LMDB record type: {type(record).__name__}")

    if not isinstance(image, bytes):
        raise ValueError(f"Unsupported LMDB image payload type: {type(image).__name__}")
    return image, int(label) if label is not None else None
