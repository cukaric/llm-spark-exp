"""Center-level scouting and sampling plans for Vec2Face+ generation."""

from __future__ import annotations

import csv
import math
import shlex
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llm_spark_exp.synthetic_faces.quality.identity_consistency import (
    ADAFACE_MODEL_ID,
    AdaFaceSample,
    score_identity_dataset,
)


@dataclass(frozen=True)
class CenterScoutThresholds:
    """Thresholds used to decide whether a center is worth scaling."""

    min_similarity: float = 0.95
    min_yield_rate: float = 0.60
    min_selected: int = 4
    min_images: int = 4


@dataclass(frozen=True)
class CenterScoutSummary:
    """One center's AdaFace stability summary."""

    identity: str
    center_index: int | None
    image_count: int
    eligible_count: int
    selected_count: int
    yield_rate: float
    similarity_min: float
    similarity_p10: float
    similarity_median: float
    similarity_mean: float
    selected_similarity_min: float | None
    selected_similarity_median: float | None
    quality_p10: float
    quality_median: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class CenterScoutReportResult:
    """Files produced by a center scout scoring run."""

    source_dir: Path
    report_dir: Path
    image_report_path: Path
    summary_report_path: Path
    identities: int
    scored_images: int
    accepted_centers: int


@dataclass(frozen=True)
class CenterVariationSchedule:
    """Vec2Face+ feature-noise schedule for a group of centers."""

    name: str
    variation_sigmas: tuple[float, ...]
    variation_weights: tuple[float, ...]

    @property
    def sigmas_text(self) -> str:
        return ",".join(format_float(value) for value in self.variation_sigmas)

    @property
    def weights_text(self) -> str:
        return ",".join(format_float(value) for value in self.variation_weights)


@dataclass(frozen=True)
class CenterSamplingPlanRow:
    """One selected center in an adaptive generation plan."""

    subset_name: str
    subset_index: int
    generated_identity: str
    center_index: int
    feature_norm: float
    norm_tier: str
    schedule: CenterVariationSchedule
    examples: int
    accepted: bool
    scout_yield_rate: float | None
    scout_similarity_p10: float | None
    scout_quality_median: float | None
    scout_reason: str


@dataclass(frozen=True)
class CenterFeatureSubset:
    """A generated center-feature subset and its original-center mapping."""

    subset_name: str
    feature_path: Path
    mapping_path: Path
    rows: int
    schedule: CenterVariationSchedule
    examples: int


LOW_NORM_SCHEDULE = CenterVariationSchedule(
    name="low_norm",
    variation_sigmas=(0.03, 0.05, 0.08),
    variation_weights=(0.50, 0.35, 0.15),
)
MID_NORM_SCHEDULE = CenterVariationSchedule(
    name="mid_norm",
    variation_sigmas=(0.04, 0.08, 0.12, 0.16),
    variation_weights=(0.45, 0.30, 0.20, 0.05),
)
HIGH_NORM_SCHEDULE = CenterVariationSchedule(
    name="high_norm",
    variation_sigmas=(0.04, 0.07, 0.11, 0.15),
    variation_weights=(0.45, 0.30, 0.20, 0.05),
)
RETRY_SCHEDULE = CenterVariationSchedule(
    name="retry",
    variation_sigmas=(0.02, 0.04, 0.06),
    variation_weights=(0.55, 0.30, 0.15),
)
DEFAULT_SCHEDULES = {
    "low": LOW_NORM_SCHEDULE,
    "mid": MID_NORM_SCHEDULE,
    "high": HIGH_NORM_SCHEDULE,
    "retry": RETRY_SCHEDULE,
}


def score_center_scout_dataset(
    *,
    source_dir: Path,
    report_dir: Path,
    keep_per_identity: int,
    thresholds: CenterScoutThresholds | None = None,
    model_id: str = ADAFACE_MODEL_ID,
    batch_size: int = 64,
    device: str = "cuda",
    anchor_count: int = 8,
    quality_weight: float = 0.05,
    diversity_weight: float = 0.02,
) -> CenterScoutReportResult:
    """Score generated candidates and write image-level plus center-level reports."""

    thresholds = thresholds or CenterScoutThresholds()
    score_result = score_identity_dataset(
        source_dir=source_dir,
        keep_per_identity=keep_per_identity,
        model_id=model_id,
        batch_size=batch_size,
        device=device,
        anchor_count=anchor_count,
        quality_weight=quality_weight,
        diversity_weight=diversity_weight,
        min_similarity=thresholds.min_similarity,
    )
    summaries = summarize_center_scout_samples(
        score_result.samples,
        thresholds=thresholds,
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    image_report_path = report_dir / "adaface_image_scores.csv"
    summary_report_path = report_dir / "center_scout_summary.csv"
    write_center_image_scores_csv(
        score_result.samples,
        image_report_path,
        min_similarity=thresholds.min_similarity,
    )
    write_center_scout_summary_csv(summaries, summary_report_path)
    return CenterScoutReportResult(
        source_dir=source_dir,
        report_dir=report_dir,
        image_report_path=image_report_path,
        summary_report_path=summary_report_path,
        identities=score_result.identities,
        scored_images=score_result.scored_images,
        accepted_centers=sum(summary.accepted for summary in summaries),
    )


def summarize_center_scout_samples(
    samples: Iterable[AdaFaceSample],
    *,
    thresholds: CenterScoutThresholds,
) -> tuple[CenterScoutSummary, ...]:
    """Summarize AdaFace samples into one decision row per center."""

    grouped: dict[str, list[AdaFaceSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.identity].append(sample)

    summaries = [
        summarize_one_center(identity, identity_samples, thresholds=thresholds)
        for identity, identity_samples in grouped.items()
    ]
    return tuple(
        sorted(
            summaries,
            key=lambda summary: (
                summary.center_index is None,
                summary.center_index if summary.center_index is not None else summary.identity,
            ),
        )
    )


def summarize_one_center(
    identity: str,
    samples: Sequence[AdaFaceSample],
    *,
    thresholds: CenterScoutThresholds,
) -> CenterScoutSummary:
    """Summarize one center's image scores and pass/fail reason."""

    similarities = np.asarray([sample.similarity for sample in samples], dtype=np.float64)
    qualities = np.asarray([sample.quality for sample in samples], dtype=np.float64)
    eligible = [sample for sample in samples if sample.similarity >= thresholds.min_similarity]
    selected = [sample for sample in samples if sample.selected]
    selected_similarities = np.asarray(
        [sample.similarity for sample in selected],
        dtype=np.float64,
    )

    image_count = len(samples)
    eligible_count = len(eligible)
    yield_rate = eligible_count / image_count if image_count else 0.0
    accepted, reason = decide_center_acceptance(
        image_count=image_count,
        eligible_count=eligible_count,
        yield_rate=yield_rate,
        thresholds=thresholds,
    )
    return CenterScoutSummary(
        identity=identity,
        center_index=parse_center_index(identity),
        image_count=image_count,
        eligible_count=eligible_count,
        selected_count=len(selected),
        yield_rate=yield_rate,
        similarity_min=float(np.min(similarities)) if image_count else math.nan,
        similarity_p10=percentile(similarities, 10),
        similarity_median=percentile(similarities, 50),
        similarity_mean=float(np.mean(similarities)) if image_count else math.nan,
        selected_similarity_min=(
            float(np.min(selected_similarities)) if len(selected_similarities) else None
        ),
        selected_similarity_median=(
            percentile(selected_similarities, 50) if len(selected_similarities) else None
        ),
        quality_p10=percentile(qualities, 10),
        quality_median=percentile(qualities, 50),
        accepted=accepted,
        reason=reason,
    )


def decide_center_acceptance(
    *,
    image_count: int,
    eligible_count: int,
    yield_rate: float,
    thresholds: CenterScoutThresholds,
) -> tuple[bool, str]:
    """Return whether a center should be scaled and the reason."""

    if image_count < thresholds.min_images:
        return False, "insufficient_images"
    if eligible_count < thresholds.min_selected:
        return False, "too_few_eligible"
    if yield_rate < thresholds.min_yield_rate:
        return False, "low_yield"
    return True, "accepted"


def build_center_sampling_plan(
    *,
    center_features: np.ndarray,
    scout_summaries: Sequence[CenterScoutSummary] = (),
    examples: int = 24,
    retry_examples: int = 12,
    include_rejected: bool = False,
    include_unscored: bool = False,
    low_norm_quantile: float = 0.15,
    high_norm_quantile: float = 0.85,
    schedules: dict[str, CenterVariationSchedule] | None = None,
) -> tuple[CenterSamplingPlanRow, ...]:
    """Build an adaptive per-center generation plan."""

    if center_features.ndim != 2:
        raise ValueError("center_features must be a 2D array.")
    if examples < 1:
        raise ValueError("examples must be at least 1.")
    if retry_examples < 1:
        raise ValueError("retry_examples must be at least 1.")
    if not 0.0 <= low_norm_quantile < high_norm_quantile <= 1.0:
        raise ValueError("norm quantiles must satisfy 0 <= low < high <= 1.")

    schedules = schedules or DEFAULT_SCHEDULES
    norms = np.linalg.norm(center_features, axis=1)
    low_cutoff, high_cutoff = np.quantile(norms, [low_norm_quantile, high_norm_quantile])
    has_scout = len(scout_summaries) > 0
    summaries_by_index = {
        summary.center_index: summary
        for summary in scout_summaries
        if summary.center_index is not None
    }
    subset_counts: dict[str, int] = defaultdict(int)
    rows: list[CenterSamplingPlanRow] = []

    for center_index, feature_norm in enumerate(norms):
        summary = summaries_by_index.get(center_index)
        if summary is None and has_scout and not include_unscored:
            continue
        if summary is not None and not summary.accepted and not include_rejected:
            continue

        accepted = summary.accepted if summary is not None else not has_scout
        norm_tier = classify_norm_tier(
            float(feature_norm),
            low_cutoff=float(low_cutoff),
            high_cutoff=float(high_cutoff),
        )
        schedule_key = norm_tier if accepted else "retry"
        schedule = schedules[schedule_key]
        planned_examples = examples if accepted else retry_examples
        subset_index = subset_counts[schedule.name]
        subset_counts[schedule.name] += 1
        rows.append(
            CenterSamplingPlanRow(
                subset_name=schedule.name,
                subset_index=subset_index,
                generated_identity=f"{subset_index:04d}",
                center_index=center_index,
                feature_norm=float(feature_norm),
                norm_tier=norm_tier,
                schedule=schedule,
                examples=planned_examples,
                accepted=accepted,
                scout_yield_rate=summary.yield_rate if summary is not None else None,
                scout_similarity_p10=summary.similarity_p10 if summary is not None else None,
                scout_quality_median=summary.quality_median if summary is not None else None,
                scout_reason=summary.reason if summary is not None else "unscored",
            )
        )

    return tuple(rows)


def export_center_feature_subsets(
    *,
    center_features: np.ndarray,
    plan_rows: Sequence[CenterSamplingPlanRow],
    output_dir: Path,
    prefix: str = "selected_centers",
) -> tuple[CenterFeatureSubset, ...]:
    """Write one feature .npy and one mapping CSV for each schedule group."""

    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[CenterSamplingPlanRow]] = defaultdict(list)
    for row in plan_rows:
        grouped[row.subset_name].append(row)

    subsets: list[CenterFeatureSubset] = []
    for subset_name, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row.subset_index)
        feature_path = output_dir / f"{prefix}_{subset_name}.npy"
        mapping_path = output_dir / f"{prefix}_{subset_name}_mapping.csv"
        subset_features = center_features[[row.center_index for row in rows]]
        np.save(feature_path, subset_features.astype(center_features.dtype, copy=False))
        write_center_sampling_plan_csv(rows, mapping_path)
        first_row = rows[0]
        subsets.append(
            CenterFeatureSubset(
                subset_name=subset_name,
                feature_path=feature_path,
                mapping_path=mapping_path,
                rows=len(rows),
                schedule=first_row.schedule,
                examples=first_row.examples,
            )
        )

    return tuple(subsets)


def write_generation_commands(
    *,
    subsets: Sequence[CenterFeatureSubset],
    output_path: Path,
    run_name_prefix: str,
    module_python: str = ".venv/bin/python",
    vec2face_python: str | None = None,
    batch_size: int = 64,
    device: str = "cuda",
) -> Path:
    """Write shell commands that run each exported feature subset."""

    vec2face_python = vec2face_python or module_python
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    for subset in subsets:
        run_name = f"{run_name_prefix}_{subset.subset_name}"
        parts = [
            "PYTHONPATH=src",
            shlex.quote(module_python),
            "-m",
            "llm_spark_exp.synthetic_faces_cli",
            "generate-from-center-features",
            "--center-feature",
            shlex.quote(str(subset.feature_path)),
            "--name",
            shlex.quote(run_name),
            "--examples",
            str(subset.examples),
            "--batch-size",
            str(batch_size),
            "--device",
            shlex.quote(device),
            "--variation-sigmas",
            shlex.quote(subset.schedule.sigmas_text),
            "--variation-weights",
            shlex.quote(subset.schedule.weights_text),
            "--python",
            shlex.quote(vec2face_python),
        ]
        lines.append(" ".join(parts))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_path.chmod(0o755)
    return output_path


def load_center_features(path: Path) -> np.ndarray:
    """Load a Vec2Face+ center-feature array and validate its shape."""

    if not path.exists():
        raise FileNotFoundError(f"Center feature file does not exist: {path}")
    features = np.load(path)
    if features.ndim != 2:
        raise ValueError(f"Center features must be a 2D array, got shape {features.shape}.")
    return features


def classify_norm_tier(
    feature_norm: float,
    *,
    low_cutoff: float,
    high_cutoff: float,
) -> str:
    """Classify a center norm into low/mid/high tier."""

    if feature_norm <= low_cutoff:
        return "low"
    if feature_norm >= high_cutoff:
        return "high"
    return "mid"


def write_center_image_scores_csv(
    samples: Iterable[AdaFaceSample],
    path: Path,
    *,
    min_similarity: float,
) -> None:
    """Write per-image AdaFace scores for auditing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_samples = sorted(
        samples,
        key=lambda sample: (
            parse_center_index(sample.identity) is None,
            parse_center_index(sample.identity)
            if parse_center_index(sample.identity) is not None
            else sample.identity,
            -sample.score,
        ),
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "identity",
                "center_index",
                "source_path",
                "similarity",
                "quality",
                "score",
                "eligible",
                "selected",
            ],
        )
        writer.writeheader()
        for sample in sorted_samples:
            writer.writerow(
                {
                    "identity": sample.identity,
                    "center_index": optional_int_text(parse_center_index(sample.identity)),
                    "source_path": str(sample.source_path),
                    "similarity": format_float(sample.similarity),
                    "quality": format_float(sample.quality),
                    "score": format_float(sample.score),
                    "eligible": str(sample.similarity >= min_similarity).lower(),
                    "selected": str(sample.selected).lower(),
                }
            )


def write_center_scout_summary_csv(
    summaries: Iterable[CenterScoutSummary],
    path: Path,
) -> None:
    """Write one center scout summary row per identity folder."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=center_scout_summary_fieldnames())
        writer.writeheader()
        for summary in summaries:
            writer.writerow(center_scout_summary_to_row(summary))


def load_center_scout_summary_csv(path: Path) -> tuple[CenterScoutSummary, ...]:
    """Read center scout summaries written by this module."""

    with path.open(newline="", encoding="utf-8") as file:
        return tuple(center_scout_summary_from_row(row) for row in csv.DictReader(file))


def write_center_sampling_plan_csv(
    rows: Iterable[CenterSamplingPlanRow],
    path: Path,
) -> None:
    """Write selected center plan rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "subset_name",
                "subset_index",
                "generated_identity",
                "center_index",
                "feature_norm",
                "norm_tier",
                "schedule",
                "examples",
                "variation_sigmas",
                "variation_weights",
                "accepted",
                "scout_yield_rate",
                "scout_similarity_p10",
                "scout_quality_median",
                "scout_reason",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "subset_name": row.subset_name,
                    "subset_index": row.subset_index,
                    "generated_identity": row.generated_identity,
                    "center_index": row.center_index,
                    "feature_norm": format_float(row.feature_norm),
                    "norm_tier": row.norm_tier,
                    "schedule": row.schedule.name,
                    "examples": row.examples,
                    "variation_sigmas": row.schedule.sigmas_text,
                    "variation_weights": row.schedule.weights_text,
                    "accepted": str(row.accepted).lower(),
                    "scout_yield_rate": optional_float_text(row.scout_yield_rate),
                    "scout_similarity_p10": optional_float_text(row.scout_similarity_p10),
                    "scout_quality_median": optional_float_text(row.scout_quality_median),
                    "scout_reason": row.scout_reason,
                }
            )


def center_scout_summary_fieldnames() -> list[str]:
    return [
        "identity",
        "center_index",
        "image_count",
        "eligible_count",
        "selected_count",
        "yield_rate",
        "similarity_min",
        "similarity_p10",
        "similarity_median",
        "similarity_mean",
        "selected_similarity_min",
        "selected_similarity_median",
        "quality_p10",
        "quality_median",
        "accepted",
        "reason",
    ]


def center_scout_summary_to_row(summary: CenterScoutSummary) -> dict[str, str]:
    return {
        "identity": summary.identity,
        "center_index": optional_int_text(summary.center_index),
        "image_count": str(summary.image_count),
        "eligible_count": str(summary.eligible_count),
        "selected_count": str(summary.selected_count),
        "yield_rate": format_float(summary.yield_rate),
        "similarity_min": format_float(summary.similarity_min),
        "similarity_p10": format_float(summary.similarity_p10),
        "similarity_median": format_float(summary.similarity_median),
        "similarity_mean": format_float(summary.similarity_mean),
        "selected_similarity_min": optional_float_text(summary.selected_similarity_min),
        "selected_similarity_median": optional_float_text(summary.selected_similarity_median),
        "quality_p10": format_float(summary.quality_p10),
        "quality_median": format_float(summary.quality_median),
        "accepted": str(summary.accepted).lower(),
        "reason": summary.reason,
    }


def center_scout_summary_from_row(row: dict[str, str]) -> CenterScoutSummary:
    return CenterScoutSummary(
        identity=row["identity"],
        center_index=optional_int(row.get("center_index")),
        image_count=int(row["image_count"]),
        eligible_count=int(row["eligible_count"]),
        selected_count=int(row["selected_count"]),
        yield_rate=float(row["yield_rate"]),
        similarity_min=float(row["similarity_min"]),
        similarity_p10=float(row["similarity_p10"]),
        similarity_median=float(row["similarity_median"]),
        similarity_mean=float(row["similarity_mean"]),
        selected_similarity_min=optional_float(row.get("selected_similarity_min")),
        selected_similarity_median=optional_float(row.get("selected_similarity_median")),
        quality_p10=float(row["quality_p10"]),
        quality_median=float(row["quality_median"]),
        accepted=row["accepted"].lower() == "true",
        reason=row["reason"],
    )


def parse_center_index(identity: str) -> int | None:
    """Parse a numeric Vec2Face+ identity folder name."""

    try:
        return int(identity)
    except ValueError:
        return None


def percentile(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return math.nan
    return float(np.percentile(values, q))


def format_float(value: float) -> str:
    return f"{value:.8g}"


def optional_float_text(value: float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return format_float(value)


def optional_int_text(value: int | None) -> str:
    return "" if value is None else str(value)


def optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)
