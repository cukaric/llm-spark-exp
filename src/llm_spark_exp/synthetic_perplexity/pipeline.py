"""Separate orchestration for a WebFace4M-class synthetic dataset build.

This module keeps the Perplexity-style 500k x 50 plan isolated from the older
``synthetic_faces`` helpers.  It deliberately focuses on deterministic planning:
center-space analysis, Gaussian center expansion, cosine-separated selection,
sharding, command generation, and manifest joins.  Expensive GPU generation and
AdaFace scoring are delegated to the existing ``synthetic-faces`` CLI.
"""

from __future__ import annotations

import csv
import json
import math
import shlex
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from llm_spark_exp.synthetic_faces.generation.vec2face_plus import (
    DEFAULT_VEC2FACE_PLUS_DIR,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
GENERATOR_SPACE_EVIDENCE = {
    "assumed_space": "ArcFace-R100 Glint360K",
    "vec2face_issue": "https://github.com/HaiyuWu/Vec2Face/issues/3#issuecomment-2374007786",
    "vec2face_hf_file": "https://huggingface.co/BooBooWu/Vec2Face/tree/main",
    "note": (
        "The upstream author says the Vec2Face features he used were extracted "
        "with ArcFace-R100 trained on Glint360K, and BooBooWu/Vec2Face publishes "
        "center_feature_examples.npy alongside arcface-r100-glint360k.pth. "
        "Custom center files should still carry their own extraction metadata."
    ),
}


@dataclass(frozen=True)
class GenerationStage:
    """One generation bucket in the 50-final-image quota."""

    name: str
    mode: Literal["center_feature", "pose_control", "attrop", "restore"]
    final_images: int
    raw_attempts: int
    variation_sigmas: tuple[float, ...] = ()
    variation_weights: tuple[float, ...] = ()
    description: str = ""

    @property
    def sigmas_text(self) -> str:
        return ",".join(format_float(value) for value in self.variation_sigmas)

    @property
    def weights_text(self) -> str:
        return ",".join(format_float(value) for value in self.variation_weights)


ANCHOR_STAGE = GenerationStage(
    name="anchor_scout",
    mode="center_feature",
    final_images=1,
    raw_attempts=3,
    variation_sigmas=(0.01, 0.02),
    variation_weights=(0.75, 0.25),
    description="Strict low-variation anchor attempts used to reject dead centers early.",
)

DEFAULT_GENERATION_STAGES = (
    GenerationStage(
        name="bulk_mild",
        mode="center_feature",
        final_images=24,
        raw_attempts=48,
        variation_sigmas=(0.04, 0.08, 0.12),
        variation_weights=(0.45, 0.35, 0.20),
        description="Main perturbation-sampling bucket for frontal and mild-yaw images.",
    ),
    GenerationStage(
        name="moderate_yaw",
        mode="pose_control",
        final_images=15,
        raw_attempts=24,
        description="LoRA landmark pose-control bucket for moderate yaw.",
    ),
    GenerationStage(
        name="large_yaw",
        mode="pose_control",
        final_images=7,
        raw_attempts=12,
        description="Lower-quota pose-control bucket for larger yaw.",
    ),
    GenerationStage(
        name="extreme_yaw",
        mode="pose_control",
        final_images=1,
        raw_attempts=3,
        description="Tiny extreme-pose quota to avoid forcing identity drift.",
    ),
    GenerationStage(
        name="other_variation",
        mode="attrop",
        final_images=3,
        raw_attempts=6,
        description="Small AttrOP-style budget for expression, lighting, and accessory variation.",
    ),
)


@dataclass(frozen=True)
class PerplexityScaleConfig:
    """Defaults for the 500k identities x 50 images plan."""

    target_identities: int = 500_000
    reserve_centers: int = 750_000
    gaussian_candidates: int = 1_500_000
    inter_center_max_cosine: float = 0.30
    final_images_per_identity: int = 50
    raw_candidates_per_identity: int = 90
    max_raw_attempts_per_identity: int = 120
    min_accepted_before_select: int = 60
    anchor_attempts: int = 3
    anchor_min_similarity: float = 0.90
    anchor_min_quality: float = 26.0
    variant_min_similarity: float = 0.70
    variant_min_quality: float = 24.0
    adaface_anchor_similarity: float = 0.95
    adaface_variant_similarity: float = 0.95
    osdface_cap_fraction: float = 0.20
    shard_size: int = 10_000
    seed: int = 20260611

    def validate(self, stages: tuple[GenerationStage, ...] = DEFAULT_GENERATION_STAGES) -> None:
        if self.target_identities < 1:
            raise ValueError("target_identities must be at least 1.")
        if self.reserve_centers < self.target_identities:
            raise ValueError("reserve_centers must be greater than or equal to target_identities.")
        if self.gaussian_candidates < 1:
            raise ValueError("gaussian_candidates must be at least 1.")
        if not 0.0 <= self.inter_center_max_cosine < 1.0:
            raise ValueError("inter_center_max_cosine must be in [0, 1).")
        if self.final_images_per_identity < 1:
            raise ValueError("final_images_per_identity must be at least 1.")
        if self.raw_candidates_per_identity < self.final_images_per_identity:
            raise ValueError("raw_candidates_per_identity must cover final_images_per_identity.")
        if self.max_raw_attempts_per_identity < self.raw_candidates_per_identity:
            raise ValueError(
                "max_raw_attempts_per_identity must cover raw_candidates_per_identity."
            )
        if self.min_accepted_before_select < self.final_images_per_identity:
            raise ValueError("min_accepted_before_select must cover final_images_per_identity.")
        if self.anchor_attempts < 1:
            raise ValueError("anchor_attempts must be at least 1.")
        if self.shard_size < 1:
            raise ValueError("shard_size must be at least 1.")
        if not 0.0 <= self.osdface_cap_fraction <= 1.0:
            raise ValueError("osdface_cap_fraction must be in [0, 1].")
        if sum(stage.final_images for stage in stages) != self.final_images_per_identity:
            raise ValueError(
                "generation stage final-image quotas must sum to final_images_per_identity."
            )

    @property
    def final_image_count(self) -> int:
        return self.target_identities * self.final_images_per_identity

    @property
    def raw_candidate_count(self) -> int:
        return self.target_identities * self.raw_candidates_per_identity

    @property
    def osdface_image_cap(self) -> int:
        return int(self.final_image_count * self.osdface_cap_fraction)


@dataclass(frozen=True)
class CenterExpansionResult:
    """Result of expanding seed centers with separated Gaussian candidates."""

    features: np.ndarray
    seed_count: int
    candidate_count: int
    accepted_count: int
    accepted_candidates: int
    max_cosine: float
    backend: str
    exhausted: bool

    def summary(self) -> dict[str, int | float | str | bool]:
        return {
            "seed_count": self.seed_count,
            "candidate_count": self.candidate_count,
            "accepted_count": self.accepted_count,
            "accepted_candidates": self.accepted_candidates,
            "max_cosine": self.max_cosine,
            "backend": self.backend,
            "exhausted": self.exhausted,
        }


@dataclass(frozen=True)
class CenterShard:
    """One feature shard and its identity mapping manifest."""

    shard_name: str
    feature_path: Path
    manifest_path: Path
    start_index: int
    end_index: int
    center_count: int


def load_center_features(path: Path) -> np.ndarray:
    """Load a 2D center-feature array."""

    if not path.exists():
        raise FileNotFoundError(f"Center feature file does not exist: {path}")
    features = np.load(path)
    if features.ndim != 2:
        raise ValueError(f"Center features must be a 2D array, got shape {features.shape}.")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("Center features must be non-empty.")
    return features


def analyze_center_features(
    center_features: np.ndarray,
    *,
    sample_size: int = 2048,
    seed: int = 20260611,
) -> dict[str, object]:
    """Return shape, norm, and sampled nearest-neighbor statistics for centers."""

    if center_features.ndim != 2:
        raise ValueError("center_features must be a 2D array.")
    norms = np.linalg.norm(center_features, axis=1)
    if np.any(norms == 0):
        raise ValueError("Center features contain zero vectors.")

    report: dict[str, object] = {
        "rows": int(center_features.shape[0]),
        "dimensions": int(center_features.shape[1]),
        "dtype": str(center_features.dtype),
        "norms": {
            "min": float(np.min(norms)),
            "p01": percentile(norms, 1),
            "p10": percentile(norms, 10),
            "median": percentile(norms, 50),
            "mean": float(np.mean(norms)),
            "p90": percentile(norms, 90),
            "p99": percentile(norms, 99),
            "max": float(np.max(norms)),
        },
        "generator_space_evidence": GENERATOR_SPACE_EVIDENCE,
    }
    report["sampled_pairwise_cosine"] = sampled_pairwise_cosine_stats(
        center_features,
        sample_size=sample_size,
        seed=seed,
    )
    return report


def sampled_pairwise_cosine_stats(
    center_features: np.ndarray,
    *,
    sample_size: int = 2048,
    seed: int = 20260611,
) -> dict[str, float | int]:
    """Estimate same-file collisions from a bounded pairwise cosine sample."""

    if len(center_features) < 2:
        return {"sample_size": len(center_features), "max": math.nan, "p99": math.nan}

    rng = np.random.default_rng(seed)
    chosen = min(sample_size, len(center_features))
    indices = rng.choice(len(center_features), size=chosen, replace=False)
    normalized = normalize_rows(center_features[indices]).astype(np.float32, copy=False)
    sims = normalized @ normalized.T
    np.fill_diagonal(sims, -np.inf)
    flattened = sims[np.isfinite(sims)]
    return {
        "sample_size": int(chosen),
        "max": float(np.max(flattened)),
        "p99": percentile(flattened, 99),
        "p95": percentile(flattened, 95),
        "mean": float(np.mean(flattened)),
    }


def sample_empirical_gaussian(
    seed_features: np.ndarray,
    *,
    count: int,
    seed: int = 20260611,
) -> np.ndarray:
    """Sample candidate centers from a diagonal Gaussian fit to seed centers.

    The vector direction is drawn from the diagonal Gaussian and the norm is
    resampled from the empirical seed norms.  That keeps candidates in the same
    rough generator-space norm regime as ``center_feature_examples.npy``.
    """

    if seed_features.ndim != 2:
        raise ValueError("seed_features must be a 2D array.")
    if count < 1:
        raise ValueError("count must be at least 1.")

    rng = np.random.default_rng(seed)
    source = seed_features.astype(np.float64, copy=False)
    mean = source.mean(axis=0)
    std = source.std(axis=0)
    std = np.where(std <= 1e-12, 1e-6, std)
    samples = rng.normal(loc=mean, scale=std, size=(count, source.shape[1])).astype(np.float32)
    seed_norms = np.linalg.norm(source, axis=1)
    if np.any(seed_norms == 0):
        raise ValueError("seed_features contain zero vectors.")
    target_norms = rng.choice(seed_norms.astype(np.float32), size=count, replace=True)
    return normalize_rows(samples) * target_norms[:, None]


def expand_center_pool(
    seed_features: np.ndarray,
    *,
    target_count: int,
    candidate_count: int,
    max_cosine: float = 0.30,
    seed: int = 20260611,
    backend: Literal["auto", "faiss", "numpy"] = "auto",
    batch_size: int = 2048,
) -> CenterExpansionResult:
    """Sample Gaussian candidates and greedily keep a separated center pool."""

    candidates = sample_empirical_gaussian(
        seed_features,
        count=candidate_count,
        seed=seed,
    )
    return greedy_accept_centers(
        seed_features=seed_features,
        candidate_features=candidates,
        target_count=target_count,
        max_cosine=max_cosine,
        backend=backend,
        batch_size=batch_size,
    )


def greedy_accept_centers(
    *,
    seed_features: np.ndarray,
    candidate_features: np.ndarray,
    target_count: int,
    max_cosine: float = 0.30,
    backend: Literal["auto", "faiss", "numpy"] = "auto",
    batch_size: int = 2048,
    numpy_max_comparisons: int = 50_000_000,
) -> CenterExpansionResult:
    """Keep seed centers first, then accept candidates with nearest cosine <= threshold."""

    validate_feature_pair(seed_features, candidate_features)
    if target_count < 1:
        raise ValueError("target_count must be at least 1.")
    if not 0.0 <= max_cosine < 1.0:
        raise ValueError("max_cosine must be in [0, 1).")

    seed_features = seed_features.astype(np.float32, copy=False)
    candidate_features = candidate_features.astype(np.float32, copy=False)
    if target_count <= len(seed_features):
        selected = seed_features[:target_count].copy()
        return CenterExpansionResult(
            features=selected,
            seed_count=len(seed_features),
            candidate_count=len(candidate_features),
            accepted_count=len(selected),
            accepted_candidates=0,
            max_cosine=max_cosine,
            backend="seed_only",
            exhausted=False,
        )

    selected_backend = choose_backend(
        backend=backend,
        seed_count=len(seed_features),
        candidate_count=len(candidate_features),
        target_count=target_count,
        numpy_max_comparisons=numpy_max_comparisons,
    )
    if selected_backend == "faiss":
        return _greedy_accept_centers_faiss(
            seed_features=seed_features,
            candidate_features=candidate_features,
            target_count=target_count,
            max_cosine=max_cosine,
            batch_size=batch_size,
        )
    return _greedy_accept_centers_numpy(
        seed_features=seed_features,
        candidate_features=candidate_features,
        target_count=target_count,
        max_cosine=max_cosine,
    )


def choose_backend(
    *,
    backend: Literal["auto", "faiss", "numpy"],
    seed_count: int,
    candidate_count: int,
    target_count: int,
    numpy_max_comparisons: int,
) -> Literal["faiss", "numpy"]:
    if backend == "faiss":
        ensure_faiss_available()
        return "faiss"
    if backend == "numpy":
        return "numpy"
    if faiss_is_available():
        return "faiss"

    rough_comparisons = (seed_count + min(candidate_count, target_count)) * candidate_count
    if rough_comparisons > numpy_max_comparisons:
        raise RuntimeError(
            "FAISS is required for this center expansion size. Install faiss-cpu/faiss-gpu "
            "or rerun with smaller --candidate-count/--target-count for a smoke test."
        )
    return "numpy"


def _greedy_accept_centers_numpy(
    *,
    seed_features: np.ndarray,
    candidate_features: np.ndarray,
    target_count: int,
    max_cosine: float,
) -> CenterExpansionResult:
    dim = seed_features.shape[1]
    selected = np.empty((target_count, dim), dtype=np.float32)
    selected_normed = np.empty((target_count, dim), dtype=np.float32)
    seed_count = min(len(seed_features), target_count)
    selected[:seed_count] = seed_features[:seed_count]
    selected_normed[:seed_count] = normalize_rows(seed_features[:seed_count])
    accepted_count = seed_count

    for candidate in candidate_features:
        if accepted_count >= target_count:
            break
        candidate_normed = normalize_rows(candidate.reshape(1, -1))[0]
        nearest = float(np.max(selected_normed[:accepted_count] @ candidate_normed))
        if nearest <= max_cosine:
            selected[accepted_count] = candidate
            selected_normed[accepted_count] = candidate_normed
            accepted_count += 1

    return CenterExpansionResult(
        features=selected[:accepted_count].copy(),
        seed_count=len(seed_features),
        candidate_count=len(candidate_features),
        accepted_count=accepted_count,
        accepted_candidates=max(0, accepted_count - seed_count),
        max_cosine=max_cosine,
        backend="numpy",
        exhausted=accepted_count < target_count,
    )


def _greedy_accept_centers_faiss(
    *,
    seed_features: np.ndarray,
    candidate_features: np.ndarray,
    target_count: int,
    max_cosine: float,
    batch_size: int,
) -> CenterExpansionResult:
    import faiss  # type: ignore[import-not-found]

    dim = seed_features.shape[1]
    index = faiss.IndexFlatIP(dim)
    seed_count = min(len(seed_features), target_count)
    accepted_chunks = [seed_features[:seed_count].copy()]
    index.add(normalize_rows(seed_features[:seed_count]).astype(np.float32))
    accepted_count = seed_count

    for start in range(0, len(candidate_features), batch_size):
        if accepted_count >= target_count:
            break
        batch = candidate_features[start : start + batch_size]
        normalized = normalize_rows(batch).astype(np.float32, copy=False)
        sims, _ = index.search(normalized, 1)
        accepted_local: list[int] = []
        pending_norms: list[np.ndarray] = []
        for local_index in np.flatnonzero(sims[:, 0] <= max_cosine):
            if accepted_count >= target_count:
                break
            query = normalized[local_index : local_index + 1]
            nearest, _ = index.search(query, 1)
            pending_nearest = -math.inf
            if pending_norms:
                pending_nearest = float(np.max(np.vstack(pending_norms) @ query[0]))
            if float(nearest[0, 0]) <= max_cosine and pending_nearest <= max_cosine:
                accepted_local.append(int(local_index))
                pending_norms.append(query[0].copy())
                accepted_count += 1
                if len(pending_norms) >= 512:
                    index.add(np.vstack(pending_norms).astype(np.float32))
                    pending_norms.clear()
        if pending_norms:
            index.add(np.vstack(pending_norms).astype(np.float32))
        if accepted_local:
            accepted_chunks.append(batch[accepted_local].copy())

    features = np.vstack(accepted_chunks)[:accepted_count]
    return CenterExpansionResult(
        features=features,
        seed_count=len(seed_features),
        candidate_count=len(candidate_features),
        accepted_count=accepted_count,
        accepted_candidates=max(0, accepted_count - seed_count),
        max_cosine=max_cosine,
        backend="faiss",
        exhausted=accepted_count < target_count,
    )


def write_anchor_plan(
    *,
    center_features_path: Path,
    output_dir: Path,
    config: PerplexityScaleConfig | None = None,
    run_name_prefix: str = "perplexity_anchor",
    module_python: str = ".venv/bin/python",
    vec2face_python: str | None = None,
    batch_size: int = 64,
    device: str = "cuda",
    generated_root: Path = DEFAULT_VEC2FACE_PLUS_DIR / "generated_images",
) -> dict[str, object]:
    """Write anchor-scout shards and command scripts for a center pool."""

    config = config or PerplexityScaleConfig()
    config.validate()
    vec2face_python = vec2face_python or module_python
    features = load_center_features(center_features_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "strategy_config.json", strategy_config_json(config))
    write_json(output_dir / "center_feature_report.json", analyze_center_features(features))

    shards = write_center_shards(
        center_features=features,
        output_dir=output_dir / "shards" / "anchor",
        prefix="anchor",
        shard_size=config.shard_size,
    )
    write_shard_index(shards, output_dir / "shards" / "anchor_shards.csv")

    command_dir = output_dir / "commands"
    command_dir.mkdir(parents=True, exist_ok=True)
    generate_lines = shell_header("Generate anchor scout images.")
    score_lines = shell_header("Score anchor scout images with the existing AdaFace scout.")
    for shard in shards:
        run_name = f"{run_name_prefix}_{shard.shard_name}"
        generate_lines.append(
            shell_join(
                [
                    "PYTHONPATH=src",
                    module_python,
                    "-m",
                    "llm_spark_exp.synthetic_faces_cli",
                    "generate-from-center-features",
                    "--center-feature",
                    str(shard.feature_path),
                    "--name",
                    run_name,
                    "--examples",
                    str(config.anchor_attempts),
                    "--batch-size",
                    str(batch_size),
                    "--device",
                    device,
                    "--variation-sigmas",
                    ANCHOR_STAGE.sigmas_text,
                    "--variation-weights",
                    ANCHOR_STAGE.weights_text,
                    "--python",
                    vec2face_python,
                ]
            )
        )
        score_lines.append(
            shell_join(
                [
                    "PYTHONPATH=src",
                    module_python,
                    "-m",
                    "llm_spark_exp.synthetic_faces_cli",
                    "score-center-scout",
                    str(generated_root / run_name),
                    str(output_dir / "reports" / "anchors" / shard.shard_name),
                    "--keep-per-identity",
                    str(config.anchor_attempts),
                    "--min-similarity",
                    str(config.adaface_anchor_similarity),
                    "--min-yield-rate",
                    "0.34",
                    "--min-selected",
                    "1",
                    "--min-images",
                    str(config.anchor_attempts),
                    "--batch-size",
                    str(batch_size),
                    "--device",
                    device,
                ]
            )
        )

    collect_lines = shell_header("Collect accepted anchor centers after scoring.")
    collect_lines.append(
        shell_join(
            [
                "PYTHONPATH=src",
                module_python,
                "-m",
                "llm_spark_exp.synthetic_perplexity_cli",
                "collect-anchor-qualifications",
                str(center_features_path),
                str(output_dir),
                "--output-feature",
                str(output_dir / "qualified" / "qualified_centers.npy"),
            ]
        )
    )

    generate_path = write_shell_script(command_dir / "01_generate_anchor_scout.sh", generate_lines)
    score_path = write_shell_script(command_dir / "02_score_anchor_scout.sh", score_lines)
    collect_path = write_shell_script(
        command_dir / "03_collect_anchor_qualifications.sh", collect_lines
    )
    return {
        "output_dir": str(output_dir),
        "center_features": str(center_features_path),
        "shards": len(shards),
        "centers": len(features),
        "commands": [str(generate_path), str(score_path), str(collect_path)],
    }


def collect_anchor_qualifications(
    *,
    center_features_path: Path,
    plan_dir: Path,
    output_feature_path: Path,
) -> dict[str, object]:
    """Join anchor scout reports back to source center indices and save accepted centers."""

    features = load_center_features(center_features_path)
    manifest_paths = sorted((plan_dir / "shards" / "anchor").glob("*.csv"))
    if not manifest_paths:
        raise FileNotFoundError(f"No anchor shard manifests found under: {plan_dir}")

    accepted_indices: list[int] = []
    report_root = plan_dir / "reports" / "anchors"
    for manifest_path in manifest_paths:
        shard_name = manifest_path.stem
        report_path = report_root / shard_name / "center_scout_summary.csv"
        if not report_path.exists():
            continue
        identity_map = read_shard_identity_map(manifest_path)
        with report_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("accepted", "").lower() != "true":
                    continue
                identity = row["identity"]
                if identity not in identity_map:
                    continue
                accepted_indices.append(identity_map[identity])

    accepted_indices = sorted(set(accepted_indices))
    output_feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_feature_path, features[accepted_indices].astype(features.dtype, copy=False))
    manifest_path = output_feature_path.with_suffix(".csv")
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["qualified_index", "source_center_index"])
        writer.writeheader()
        for qualified_index, source_index in enumerate(accepted_indices):
            writer.writerow(
                {
                    "qualified_index": qualified_index,
                    "source_center_index": source_index,
                }
            )
    summary = {
        "source_centers": len(features),
        "accepted_centers": len(accepted_indices),
        "output_feature": str(output_feature_path),
        "manifest": str(manifest_path),
    }
    write_json(output_feature_path.with_suffix(".summary.json"), summary)
    return summary


def write_production_plan(
    *,
    center_features_path: Path,
    output_dir: Path,
    config: PerplexityScaleConfig | None = None,
    stages: tuple[GenerationStage, ...] = DEFAULT_GENERATION_STAGES,
    run_name_prefix: str = "perplexity_prod",
    module_python: str = ".venv/bin/python",
    vec2face_python: str | None = None,
    batch_size: int = 64,
    device: str = "cuda",
    generated_root: Path = DEFAULT_VEC2FACE_PLUS_DIR / "generated_images",
) -> dict[str, object]:
    """Write production shards, stage manifests, and command scripts."""

    config = config or PerplexityScaleConfig()
    config.validate(stages=stages)
    vec2face_python = vec2face_python or module_python
    features = load_center_features(center_features_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "strategy_config.json", strategy_config_json(config, stages=stages))
    write_json(output_dir / "center_feature_report.json", analyze_center_features(features))

    shards = write_center_shards(
        center_features=features,
        output_dir=output_dir / "shards" / "production",
        prefix="production",
        shard_size=config.shard_size,
    )
    write_shard_index(shards, output_dir / "shards" / "production_shards.csv")
    write_stage_manifest(stages, output_dir / "production_stage_manifest.csv")

    runs_path = output_dir / "production_runs.csv"
    command_dir = output_dir / "commands"
    command_dir.mkdir(parents=True, exist_ok=True)
    generate_lines = shell_header("Generate production center-feature stages.")
    pose_lines = shell_header("Pose/AttrOP stages need anchor-reference wiring before execution.")
    run_rows: list[dict[str, str]] = []
    for stage in stages:
        for shard in shards:
            run_name = f"{run_name_prefix}_{stage.name}_{shard.shard_name}"
            run_rows.append(
                {
                    "stage_name": stage.name,
                    "mode": stage.mode,
                    "shard_name": shard.shard_name,
                    "run_name": run_name,
                    "feature_path": str(shard.feature_path),
                    "manifest_path": str(shard.manifest_path),
                    "generated_dir": str(generated_root / run_name),
                    "final_images": str(stage.final_images),
                    "raw_attempts": str(stage.raw_attempts),
                }
            )
            if stage.mode == "center_feature":
                generate_lines.append(
                    shell_join(
                        [
                            "PYTHONPATH=src",
                            module_python,
                            "-m",
                            "llm_spark_exp.synthetic_faces_cli",
                            "generate-from-center-features",
                            "--center-feature",
                            str(shard.feature_path),
                            "--name",
                            run_name,
                            "--examples",
                            str(stage.raw_attempts),
                            "--batch-size",
                            str(batch_size),
                            "--device",
                            device,
                            "--variation-sigmas",
                            stage.sigmas_text,
                            "--variation-weights",
                            stage.weights_text,
                            "--python",
                            vec2face_python,
                        ]
                    )
                )
            else:
                pose_lines.append(
                    "# "
                    + shell_join(
                        [
                            stage.mode,
                            stage.name,
                            shard.shard_name,
                            "requires per-identity anchor image mapping before command emission",
                        ]
                    )
                )

    write_dict_rows(runs_path, run_rows)
    generate_path = write_shell_script(
        command_dir / "04_generate_production_center_features.sh", generate_lines
    )
    pose_path = write_shell_script(command_dir / "04b_pose_attrop_stage_manifest.sh", pose_lines)

    merge_lines = shell_header("Merge generated center-feature stage folders by source identity.")
    merge_lines.append(
        shell_join(
            [
                "PYTHONPATH=src",
                module_python,
                "-m",
                "llm_spark_exp.synthetic_perplexity_cli",
                "merge-center-feature-runs",
                str(output_dir),
                str(output_dir / "candidates" / "merged_center_feature"),
                "--generated-root",
                str(generated_root),
            ]
        )
    )
    merge_path = write_shell_script(
        command_dir / "05_merge_center_feature_candidates.sh", merge_lines
    )

    filter_lines = shell_header("Select top 50 per identity from merged candidates with AdaFace.")
    filter_lines.append(
        shell_join(
            [
                "PYTHONPATH=src",
                module_python,
                "-m",
                "llm_spark_exp.synthetic_faces_cli",
                "filter-with-adaface",
                str(output_dir / "candidates" / "merged_center_feature"),
                str(output_dir / "final" / "adaface_top50"),
                "--keep-per-identity",
                str(config.final_images_per_identity),
                "--min-similarity",
                str(config.adaface_variant_similarity),
                "--batch-size",
                str(batch_size),
                "--device",
                device,
                "--diversity-weight",
                "0.02",
                "--quality-weight",
                "0.05",
            ]
        )
    )
    filter_path = write_shell_script(command_dir / "06_filter_top50_adaface.sh", filter_lines)

    return {
        "output_dir": str(output_dir),
        "center_features": str(center_features_path),
        "centers": len(features),
        "shards": len(shards),
        "runs": len(run_rows),
        "center_feature_runs": sum(row["mode"] == "center_feature" for row in run_rows),
        "commands": [str(generate_path), str(pose_path), str(merge_path), str(filter_path)],
        "production_runs": str(runs_path),
    }


def merge_center_feature_runs(
    *,
    plan_dir: Path,
    output_dir: Path,
    generated_root: Path = DEFAULT_VEC2FACE_PLUS_DIR / "generated_images",
    copy_files: bool = False,
) -> dict[str, object]:
    """Merge generated center-feature runs into one source-indexed candidate tree."""

    runs_path = plan_dir / "production_runs.csv"
    if not runs_path.exists():
        raise FileNotFoundError(f"Missing production run manifest: {runs_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    identities: set[str] = set()
    for row in read_dict_rows(runs_path):
        if row["mode"] != "center_feature":
            continue
        run_dir = generated_root / row["run_name"]
        if not run_dir.exists():
            continue
        identity_map = read_shard_identity_map(Path(row["manifest_path"]))
        for identity_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
            source_index = identity_map.get(identity_dir.name)
            if source_index is None:
                continue
            target_identity = f"{source_index:06d}"
            target_dir = output_dir / target_identity
            target_dir.mkdir(parents=True, exist_ok=True)
            identities.add(target_identity)
            for image_path in sorted(
                path
                for path in identity_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ):
                target_name = (
                    f"{row['stage_name']}__{row['shard_name']}__"
                    f"{identity_dir.name}__{image_path.name}"
                )
                target_path = target_dir / target_name
                if target_path.exists():
                    continue
                if copy_files:
                    shutil.copy2(image_path, target_path)
                else:
                    target_path.symlink_to(image_path.resolve())
                copied += 1

    summary = {
        "output_dir": str(output_dir),
        "identities": len(identities),
        "linked_or_copied_images": copied,
        "copy_files": copy_files,
    }
    write_json(output_dir / "merge_summary.json", summary)
    return summary


def write_center_shards(
    *,
    center_features: np.ndarray,
    output_dir: Path,
    prefix: str,
    shard_size: int,
) -> tuple[CenterShard, ...]:
    """Write feature shards and CSV manifests mapping generated IDs to source indices."""

    if center_features.ndim != 2:
        raise ValueError("center_features must be a 2D array.")
    if shard_size < 1:
        raise ValueError("shard_size must be at least 1.")
    output_dir.mkdir(parents=True, exist_ok=True)
    shards: list[CenterShard] = []
    width = max(4, len(str(shard_size - 1)))
    for shard_index, start in enumerate(range(0, len(center_features), shard_size)):
        end = min(start + shard_size, len(center_features))
        shard_name = f"{prefix}_{shard_index:05d}"
        feature_path = output_dir / f"{shard_name}.npy"
        manifest_path = output_dir / f"{shard_name}.csv"
        np.save(feature_path, center_features[start:end].astype(center_features.dtype, copy=False))
        with manifest_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "shard_name",
                    "shard_row",
                    "generated_identity",
                    "source_center_index",
                ],
            )
            writer.writeheader()
            for shard_row, source_index in enumerate(range(start, end)):
                writer.writerow(
                    {
                        "shard_name": shard_name,
                        "shard_row": shard_row,
                        "generated_identity": f"{shard_row:0{width}d}",
                        "source_center_index": source_index,
                    }
                )
        shards.append(
            CenterShard(
                shard_name=shard_name,
                feature_path=feature_path,
                manifest_path=manifest_path,
                start_index=start,
                end_index=end,
                center_count=end - start,
            )
        )
    return tuple(shards)


def write_shard_index(shards: tuple[CenterShard, ...], path: Path) -> None:
    rows = [
        {
            "shard_name": shard.shard_name,
            "feature_path": str(shard.feature_path),
            "manifest_path": str(shard.manifest_path),
            "start_index": str(shard.start_index),
            "end_index": str(shard.end_index),
            "center_count": str(shard.center_count),
        }
        for shard in shards
    ]
    write_dict_rows(path, rows)


def write_stage_manifest(stages: tuple[GenerationStage, ...], path: Path) -> None:
    rows = [
        {
            "stage_name": stage.name,
            "mode": stage.mode,
            "final_images": str(stage.final_images),
            "raw_attempts": str(stage.raw_attempts),
            "variation_sigmas": stage.sigmas_text,
            "variation_weights": stage.weights_text,
            "description": stage.description,
        }
        for stage in stages
    ]
    write_dict_rows(path, rows)


def read_shard_identity_map(manifest_path: Path) -> dict[str, int]:
    with manifest_path.open(newline="", encoding="utf-8") as file:
        return {
            row["generated_identity"]: int(row["source_center_index"])
            for row in csv.DictReader(file)
        }


def strategy_config_json(
    config: PerplexityScaleConfig,
    *,
    stages: tuple[GenerationStage, ...] = DEFAULT_GENERATION_STAGES,
) -> dict[str, object]:
    return {
        "config": asdict(config),
        "scale_estimates": {
            "final_images": config.final_image_count,
            "raw_candidates": config.raw_candidate_count,
            "osdface_final_image_cap": config.osdface_image_cap,
        },
        "anchor_stage": stage_to_dict(ANCHOR_STAGE),
        "generation_stages": [stage_to_dict(stage) for stage in stages],
        "validator_gaps": [
            "Generator-space ArcFace post-generation gates are not implemented in this package yet.",
            "AuraFace validation is reserved as a downstream cross-space validator.",
            "Pose-control and AttrOP stages are represented as manifests until anchor image wiring exists.",
        ],
    }


def stage_to_dict(stage: GenerationStage) -> dict[str, object]:
    row = asdict(stage)
    row["variation_sigmas"] = list(stage.variation_sigmas)
    row["variation_weights"] = list(stage.variation_weights)
    return row


def validate_feature_pair(seed_features: np.ndarray, candidate_features: np.ndarray) -> None:
    if seed_features.ndim != 2 or candidate_features.ndim != 2:
        raise ValueError("seed_features and candidate_features must be 2D arrays.")
    if seed_features.shape[1] != candidate_features.shape[1]:
        raise ValueError("seed_features and candidate_features must have the same dimension.")
    if len(seed_features) == 0:
        raise ValueError("seed_features must be non-empty.")
    if len(candidate_features) == 0:
        raise ValueError("candidate_features must be non-empty.")


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize zero vectors.")
    return values / norms


def faiss_is_available() -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


def ensure_faiss_available() -> None:
    if not faiss_is_available():
        raise RuntimeError("FAISS is not installed. Install faiss-cpu or faiss-gpu.")


def percentile(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return math.nan
    return float(np.percentile(values, q))


def format_float(value: float) -> str:
    return f"{value:.8g}"


def shell_header(description: str) -> list[str]:
    return ["#!/usr/bin/env bash", "set -euo pipefail", "", f"# {description}"]


def shell_join(parts: list[str]) -> str:
    if not parts:
        return ""
    rendered = [parts[0]] + [shlex.quote(part) for part in parts[1:]]
    return " ".join(rendered)


def write_shell_script(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_dict_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))
