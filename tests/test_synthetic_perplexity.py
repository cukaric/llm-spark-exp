import csv
from pathlib import Path

import numpy as np

from llm_spark_exp.synthetic_perplexity import (
    DEFAULT_GENERATION_STAGES,
    PerplexityScaleConfig,
    analyze_center_features,
    collect_anchor_qualifications,
    greedy_accept_centers,
    sample_empirical_gaussian,
    write_anchor_plan,
    write_production_plan,
)


def test_perplexity_config_matches_50_image_quota() -> None:
    config = PerplexityScaleConfig(target_identities=2, reserve_centers=3)

    config.validate()

    assert sum(stage.final_images for stage in DEFAULT_GENERATION_STAGES) == 50
    assert config.final_image_count == 100
    assert config.raw_candidate_count == 180
    assert config.osdface_image_cap == 20


def test_analyze_center_features_reports_generator_space_evidence() -> None:
    features = np.eye(4, dtype=np.float32)

    report = analyze_center_features(features, sample_size=4)

    assert report["rows"] == 4
    assert report["dimensions"] == 4
    assert report["generator_space_evidence"]["assumed_space"] == "ArcFace-R100 Glint360K"
    assert report["sampled_pairwise_cosine"]["max"] == 0.0


def test_sample_empirical_gaussian_preserves_seed_norm_regime() -> None:
    seed_features = np.array(
        [
            [2.0, 0.0],
            [0.0, 3.0],
            [-4.0, 0.0],
        ],
        dtype=np.float32,
    )

    samples = sample_empirical_gaussian(seed_features, count=12, seed=7)
    sample_norms = np.linalg.norm(samples, axis=1)

    assert samples.shape == (12, 2)
    assert set(np.round(sample_norms, 5)).issubset({2.0, 3.0, 4.0})


def test_greedy_accept_centers_rejects_cosine_collision() -> None:
    seed_features = np.array([[1.0, 0.0]], dtype=np.float32)
    candidates = np.array(
        [
            [0.99, 0.01],
            [0.0, 1.0],
            [0.1, 0.99],
        ],
        dtype=np.float32,
    )

    result = greedy_accept_centers(
        seed_features=seed_features,
        candidate_features=candidates,
        target_count=2,
        max_cosine=0.30,
        backend="numpy",
    )

    assert result.accepted_count == 2
    assert result.accepted_candidates == 1
    np.testing.assert_allclose(result.features[1], [0.0, 1.0])


def test_write_anchor_plan_and_collect_qualifications(tmp_path: Path) -> None:
    features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    center_path = tmp_path / "centers.npy"
    np.save(center_path, features)
    config = PerplexityScaleConfig(target_identities=2, reserve_centers=3, shard_size=2)

    summary = write_anchor_plan(
        center_features_path=center_path,
        output_dir=tmp_path / "plan",
        config=config,
        module_python="python",
        batch_size=2,
        device="cpu",
    )

    assert summary["shards"] == 2
    command_text = (tmp_path / "plan" / "commands" / "02_score_anchor_scout.sh").read_text(
        encoding="utf-8"
    )
    assert "score-center-scout" in command_text
    assert "--min-selected 1" in command_text

    write_center_scout_report(
        tmp_path / "plan" / "reports" / "anchors" / "anchor_00000" / "center_scout_summary.csv",
        [("0000", True), ("0001", False)],
    )
    write_center_scout_report(
        tmp_path / "plan" / "reports" / "anchors" / "anchor_00001" / "center_scout_summary.csv",
        [("0000", True)],
    )

    collect_summary = collect_anchor_qualifications(
        center_features_path=center_path,
        plan_dir=tmp_path / "plan",
        output_feature_path=tmp_path / "plan" / "qualified" / "qualified.npy",
    )

    qualified = np.load(tmp_path / "plan" / "qualified" / "qualified.npy")
    assert collect_summary["accepted_centers"] == 2
    np.testing.assert_allclose(qualified, features[[0, 2]])


def test_write_production_plan_marks_pose_stages_as_manifests(tmp_path: Path) -> None:
    features = np.eye(3, dtype=np.float32)
    center_path = tmp_path / "qualified.npy"
    np.save(center_path, features)
    config = PerplexityScaleConfig(target_identities=2, reserve_centers=3, shard_size=2)

    summary = write_production_plan(
        center_features_path=center_path,
        output_dir=tmp_path / "prod",
        config=config,
        module_python="python",
        batch_size=2,
        device="cpu",
    )

    assert summary["shards"] == 2
    assert summary["center_feature_runs"] == 2
    generate_text = (
        tmp_path / "prod" / "commands" / "04_generate_production_center_features.sh"
    ).read_text(encoding="utf-8")
    pose_text = (tmp_path / "prod" / "commands" / "04b_pose_attrop_stage_manifest.sh").read_text(
        encoding="utf-8"
    )
    assert "generate-from-center-features" in generate_text
    assert "requires per-identity anchor image mapping" in pose_text


def write_center_scout_report(path: Path, rows: list[tuple[str, bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for identity, accepted in rows:
            writer.writerow(
                {
                    "identity": identity,
                    "center_index": int(identity),
                    "image_count": 3,
                    "eligible_count": 1,
                    "selected_count": 1,
                    "yield_rate": 0.34,
                    "similarity_min": 0.95,
                    "similarity_p10": 0.95,
                    "similarity_median": 0.96,
                    "similarity_mean": 0.96,
                    "selected_similarity_min": 0.96,
                    "selected_similarity_median": 0.96,
                    "quality_p10": 1.0,
                    "quality_median": 1.0,
                    "accepted": str(accepted).lower(),
                    "reason": "accepted" if accepted else "too_few_eligible",
                }
            )
