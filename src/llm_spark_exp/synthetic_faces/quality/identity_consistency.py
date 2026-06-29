"""AdaFace-based identity consistency filtering for synthetic face datasets."""

from __future__ import annotations

import csv
import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from llm_spark_exp.paths import MODELS_DIR
from llm_spark_exp.synthetic_faces._common import collect_identity_images

ADAFACE_MODEL_ID = "minchul/cvlface_adaface_ir50_ms1mv2"
DEFAULT_ADAFACE_CACHE_DIR = MODELS_DIR / "adaface"


@dataclass(frozen=True)
class AdaFaceSample:
    """One image scored by the AdaFace filter."""

    identity: str
    source_path: Path
    similarity: float
    quality: float
    score: float
    selected: bool = False


@dataclass(frozen=True)
class AdaFaceFilterResult:
    """Summary of a completed AdaFace filtering run."""

    source_dir: Path
    output_dir: Path
    report_path: Path
    identities: int
    copied_images: int
    scored_images: int


@dataclass(frozen=True)
class AdaFaceScoreResult:
    """Identity-organized AdaFace scores before any files are copied."""

    source_dir: Path
    identities: int
    scored_images: int
    samples: tuple[AdaFaceSample, ...]


def filter_identity_dataset(
    *,
    source_dir: Path,
    output_dir: Path,
    keep_per_identity: int,
    model_id: str = ADAFACE_MODEL_ID,
    batch_size: int = 64,
    device: str = "cuda",
    anchor_count: int = 8,
    quality_weight: float = 0.05,
    diversity_weight: float = 0.0,
    min_similarity: float | None = None,
    report_path: Path | None = None,
) -> AdaFaceFilterResult:
    """Filter a directory of identity folders with AdaFace embeddings."""

    if keep_per_identity < 1:
        raise ValueError("keep_per_identity must be at least 1.")

    score_result = score_identity_dataset(
        source_dir=source_dir,
        keep_per_identity=keep_per_identity,
        model_id=model_id,
        batch_size=batch_size,
        device=device,
        anchor_count=anchor_count,
        quality_weight=quality_weight,
        diversity_weight=diversity_weight,
        min_similarity=min_similarity,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_path or output_dir / "adaface_filter_report.csv"
    selected = [sample for sample in score_result.samples if sample.selected]
    copy_selected_samples(selected, output_dir=output_dir)
    write_filter_report(report_path, list(score_result.samples))
    return AdaFaceFilterResult(
        source_dir=source_dir,
        output_dir=output_dir,
        report_path=report_path,
        identities=score_result.identities,
        copied_images=len(selected),
        scored_images=score_result.scored_images,
    )


def score_identity_dataset(
    *,
    source_dir: Path,
    keep_per_identity: int,
    model_id: str = ADAFACE_MODEL_ID,
    batch_size: int = 64,
    device: str = "cuda",
    anchor_count: int = 8,
    quality_weight: float = 0.05,
    diversity_weight: float = 0.0,
    min_similarity: float | None = None,
) -> AdaFaceScoreResult:
    """Score a directory of identity folders with AdaFace without copying files."""

    if keep_per_identity < 1:
        raise ValueError("keep_per_identity must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if anchor_count < 1:
        raise ValueError("anchor_count must be at least 1.")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    identities = collect_identity_images(source_dir)
    if not identities:
        raise ValueError(f"No identity image folders found in: {source_dir}")

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    model = load_adaface_model(model_id=model_id, device=device)

    all_rows: list[AdaFaceSample] = []
    for identity, image_paths in identities.items():
        embeddings, qualities = embed_images(
            image_paths,
            model=model,
            batch_size=batch_size,
            device=device,
        )
        ranked = rank_identity_samples(
            identity=identity,
            image_paths=image_paths,
            embeddings=embeddings,
            qualities=qualities,
            keep_per_identity=keep_per_identity,
            anchor_count=anchor_count,
            quality_weight=quality_weight,
            diversity_weight=diversity_weight,
            min_similarity=min_similarity,
        )
        all_rows.extend(ranked)

    return AdaFaceScoreResult(
        source_dir=source_dir,
        identities=len(identities),
        scored_images=len(all_rows),
        samples=tuple(all_rows),
    )


def load_adaface_model(*, model_id: str = ADAFACE_MODEL_ID, device: str = "cuda") -> Any:
    """Load the Hugging Face CVLFace AdaFace model."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required for AdaFace filtering.") from error

    local_dir = DEFAULT_ADAFACE_CACHE_DIR / model_id.replace("/", "__")
    local_dir.mkdir(parents=True, exist_ok=True)
    files_path = hf_hub_download(model_id, "files.txt", local_dir=str(local_dir))
    files = Path(files_path).read_text(encoding="utf-8").splitlines()
    for filename in [file for file in files if file] + [
        "config.json",
        "wrapper.py",
        "model.safetensors",
    ]:
        if not (local_dir / filename).exists():
            hf_hub_download(model_id, filename, local_dir=str(local_dir))

    cwd = os.getcwd()
    old_models_module = sys.modules.pop("models", None)
    sys.path.insert(0, str(local_dir))
    try:
        os.chdir(local_dir)
        spec = importlib.util.spec_from_file_location(
            "_llm_spark_adaface_wrapper", local_dir / "wrapper.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load AdaFace wrapper from: {local_dir}")
        wrapper = importlib.util.module_from_spec(spec)
        sys.modules["_llm_spark_adaface_wrapper"] = wrapper
        spec.loader.exec_module(wrapper)
        model = wrapper.CVLFaceRecognitionModel(wrapper.ModelConfig())
    finally:
        os.chdir(cwd)
        sys.path.pop(0)
        if old_models_module is not None:
            sys.modules["models"] = old_models_module

    model.eval()
    model.to(device)
    return model


def embed_images(
    image_paths: tuple[Path, ...],
    *,
    model: Any,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed images with AdaFace and return normalized embeddings plus quality norms."""

    import torch

    embeddings: list[np.ndarray] = []
    qualities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            inputs = torch.stack([load_adaface_input(path) for path in batch_paths]).to(device)
            output = model(inputs)
            features, norms = parse_adaface_output(output)
            if norms is None:
                norms = torch.linalg.vector_norm(features, ord=2, dim=1)
            normalized = torch.nn.functional.normalize(features.float(), p=2, dim=1)
            embeddings.append(normalized.detach().cpu().numpy())
            qualities.append(norms.float().reshape(-1).detach().cpu().numpy())
    return np.vstack(embeddings), np.concatenate(qualities)


def load_adaface_input(path: Path) -> Any:
    """Load one aligned face crop as normalized RGB tensor."""

    import torch

    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((112, 112), Image.Resampling.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    normalized = (array - 0.5) / 0.5
    return torch.from_numpy(normalized.transpose(2, 0, 1)).float()


def parse_adaface_output(output: Any) -> tuple[Any, Any | None]:
    """Return feature tensor and optional norm tensor from CVLFace/AdaFace outputs."""

    if isinstance(output, tuple):
        features = output[0]
        norms = output[1] if len(output) > 1 else None
        return features, norms
    if isinstance(output, dict):
        features = output.get("features") or output.get("last_hidden_state")
        norms = output.get("norms") or output.get("feature_norm")
        if features is None:
            raise ValueError("AdaFace model output did not include features.")
        return features, norms
    if hasattr(output, "features"):
        return output.features, getattr(output, "norms", None)
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state, getattr(output, "norms", None)
    return output, None


def rank_identity_samples(
    *,
    identity: str,
    image_paths: tuple[Path, ...],
    embeddings: np.ndarray,
    qualities: np.ndarray,
    keep_per_identity: int,
    anchor_count: int = 8,
    quality_weight: float = 0.05,
    diversity_weight: float = 0.0,
    min_similarity: float | None = None,
) -> list[AdaFaceSample]:
    """Rank samples for one identity using centroid similarity and quality norm."""

    if len(image_paths) != len(embeddings) or len(image_paths) != len(qualities):
        raise ValueError("image_paths, embeddings, and qualities must have the same length.")
    if len(image_paths) == 0:
        return []

    initial_centroid = normalize_vector(embeddings.mean(axis=0))
    initial_sims = embeddings @ initial_centroid
    top_anchor_indices = np.argsort(initial_sims)[::-1][: min(anchor_count, len(image_paths))]
    centroid = normalize_vector(embeddings[top_anchor_indices].mean(axis=0))
    similarities = embeddings @ centroid
    quality_zscores = zscore(qualities)
    scores = similarities + quality_weight * quality_zscores
    eligible = np.arange(len(image_paths))
    if min_similarity is not None:
        eligible = eligible[similarities >= min_similarity]
    selected_indices = set(
        select_diverse_indices(
            embeddings=embeddings,
            scores=scores,
            eligible=eligible,
            keep=keep_per_identity,
            diversity_weight=diversity_weight,
        )
    )

    samples = [
        AdaFaceSample(
            identity=identity,
            source_path=image_path,
            similarity=float(similarities[index]),
            quality=float(qualities[index]),
            score=float(scores[index]),
            selected=index in selected_indices,
        )
        for index, image_path in enumerate(image_paths)
    ]
    return sorted(samples, key=lambda sample: sample.score, reverse=True)


def select_diverse_indices(
    *,
    embeddings: np.ndarray,
    scores: np.ndarray,
    eligible: np.ndarray,
    keep: int,
    diversity_weight: float = 0.0,
) -> list[int]:
    """Select high-scoring samples while penalizing near-duplicates."""

    if len(eligible) == 0 or keep < 1:
        return []
    if diversity_weight <= 0:
        return eligible[np.argsort(scores[eligible])[::-1][:keep]].tolist()

    remaining = set(eligible.tolist())
    selected: list[int] = []
    while remaining and len(selected) < keep:
        best_index = None
        best_score = -np.inf
        for index in remaining:
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = float(np.max(embeddings[index] @ embeddings[selected].T))
            adjusted_score = float(scores[index]) - diversity_weight * diversity_penalty
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
    return selected


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def zscore(values: np.ndarray) -> np.ndarray:
    std = values.std()
    if std == 0:
        return np.zeros_like(values, dtype=np.float32)
    return (values - values.mean()) / std


def copy_selected_samples(samples: list[AdaFaceSample], *, output_dir: Path) -> None:
    """Copy selected samples into identity folders with dense numbering."""

    by_identity: dict[str, list[AdaFaceSample]] = {}
    for sample in samples:
        by_identity.setdefault(sample.identity, []).append(sample)

    for identity, identity_samples in by_identity.items():
        identity_dir = output_dir / identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(
            sorted(identity_samples, key=lambda item: item.score, reverse=True)
        ):
            destination = identity_dir / f"{index:03d}{sample.source_path.suffix.lower()}"
            shutil.copy2(sample.source_path, destination)


def write_filter_report(path: Path, samples: list[AdaFaceSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "identity",
                "source_path",
                "similarity",
                "quality",
                "score",
                "selected",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "identity": sample.identity,
                    "source_path": str(sample.source_path),
                    "similarity": f"{sample.similarity:.8f}",
                    "quality": f"{sample.quality:.8f}",
                    "score": f"{sample.score:.8f}",
                    "selected": str(sample.selected).lower(),
                }
            )
