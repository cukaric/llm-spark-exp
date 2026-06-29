from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from llm_spark_exp.synthetic_faces import (
    HF_VEC2FACE_PLUS_REPO_ID,
    HSFACE_DATASETS,
    VFACE_DATASETS,
    AdaFaceSample,
    CenterScoutSummary,
    CenterScoutThresholds,
    Vec2FacePlusPaths,
    Vec2FacePlusRunner,
    augment_identity_dataset,
    augment_pose_identity_dataset,
    build_center_feature_generation_command,
    build_center_sampling_plan,
    build_pose_generation_command,
    default_vec2face_plus_weights,
    download_hsface_lmdb,
    download_vface_lmdb,
    export_center_feature_subsets,
    rank_identity_samples,
    summarize_center_scout_samples,
    write_generation_commands,
)
from llm_spark_exp.synthetic_faces.generation.vec2face_plus import (
    build_huggingface_download_commands,
)
from llm_spark_exp.synthetic_faces.quality.identity_consistency import select_diverse_indices


def test_build_pose_generation_command_matches_vec2face_plus_script() -> None:
    command = build_pose_generation_command(
        image_file=Path("references/aligned.png"),
        pose_file=Path("landmarks/landmark_examples.txt"),
        model_weights=Path("vec2face_plus/pose_model.pth"),
        name="test_examples",
    )

    assert command == (
        "python",
        "pose_control.py",
        "--batch_size",
        "64",
        "--image_file",
        "references/aligned.png",
        "--example",
        "1",
        "--name",
        "test_examples",
        "--model_weights",
        "vec2face_plus/pose_model.pth",
        "--pose_file",
        "landmarks/landmark_examples.txt",
        "--use_lora",
        "--lora_r",
        "8",
    )


def test_build_pose_generation_command_can_disable_lora() -> None:
    command = build_pose_generation_command(
        image_file=Path("aligned.png"),
        pose_file=Path("landmarks.txt"),
        model_weights=Path("pose_model.pth"),
        name="without_lora",
        use_lora=False,
    )

    assert "--use_lora" not in command
    assert "--lora_r" not in command


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"examples": 0}, "examples"),
        ({"lora_r": 0}, "lora_r"),
        ({"name": " "}, "name"),
    ],
)
def test_build_pose_generation_command_validates_options(kwargs, message) -> None:
    base_kwargs = {
        "image_file": Path("aligned.png"),
        "pose_file": Path("landmarks.txt"),
        "model_weights": Path("pose_model.pth"),
        "name": "faces",
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        build_pose_generation_command(**base_kwargs)


def test_vec2face_plus_runner_uses_configured_paths(tmp_path) -> None:
    paths = Vec2FacePlusPaths(
        repo_dir=tmp_path / "Vec2Face_plus",
        weights_dir=tmp_path / "weights",
        python_executable="python3",
    )
    runner = Vec2FacePlusRunner(paths)

    command = runner.build_pose_command(
        image_file=tmp_path / "aligned.png",
        pose_file=tmp_path / "landmarks.txt",
        name="faces",
        batch_size=2,
        examples=3,
    )

    assert command[:2] == ("python3", "pose_control.py")
    assert str(paths.pose_model_weights) in command
    assert "2" in command
    assert "3" in command
    expected_output_dir = tmp_path / "Vec2Face_plus" / "generated_images_ref" / "faces"
    assert paths.generated_images_dir("faces") == expected_output_dir


def test_default_vec2face_plus_weights_include_pose_model(tmp_path) -> None:
    weights = default_vec2face_plus_weights(weights_dir=tmp_path)

    assert any(weight.repo_id == HF_VEC2FACE_PLUS_REPO_ID for weight in weights)
    assert any(weight.filename == "vec2face_plus/pose_model.pth" for weight in weights)
    assert all(weight.local_dir == tmp_path for weight in weights)


def test_build_huggingface_download_commands_returns_python_snippets(tmp_path) -> None:
    commands = build_huggingface_download_commands(
        default_vec2face_plus_weights(weights_dir=tmp_path)
    )

    assert len(commands) == 5
    assert all("hf_hub_download" in command for command in commands)
    assert any("vec2face_plus/main_model.pth" in command for command in commands)


def test_build_center_feature_generation_command() -> None:
    command = build_center_feature_generation_command(
        repo_dir=Path("models/vec2face_plus/repo"),
        center_feature=Path("models/vec2face_plus/center_feature_examples.npy"),
        model_weights=Path("models/vec2face_plus/repo/vec2face_plus/main_model.pth"),
        name="new_identities",
        batch_size=8,
        examples=2,
        start_end="0:10",
        python_executable="python3",
        skip_feature_refinement=True,
        variation_sigmas="0.05,0.1",
        variation_weights="0.7,0.3",
    )

    assert command[:3] == (
        "python3",
        "-m",
        "llm_spark_exp.synthetic_faces.generation.center_worker",
    )
    assert "--center-feature" in command
    assert "0:10" in command
    assert "--skip-feature-refinement" in command
    assert "0.05,0.1" in command
    assert "0.7,0.3" in command


def test_vec2face_plus_runner_builds_center_feature_command(tmp_path) -> None:
    paths = Vec2FacePlusPaths(
        repo_dir=tmp_path / "Vec2Face_plus",
        weights_dir=tmp_path / "weights",
        python_executable="python3",
    )
    runner = Vec2FacePlusRunner(paths)

    command = runner.build_center_feature_command(
        center_feature=tmp_path / "center_feature_examples.npy",
        name="ids",
        start_end="0:5",
    )

    assert command[:3] == (
        "python3",
        "-m",
        "llm_spark_exp.synthetic_faces.generation.center_worker",
    )
    assert str(paths.main_model_weights) in command
    assert (
        paths.generated_center_images_dir("ids")
        == tmp_path / "Vec2Face_plus" / ("generated_images") / "ids"
    )


def test_download_vface_lmdb_rejects_unknown_size(tmp_path) -> None:
    with pytest.raises(ValueError, match="dataset_size"):
        download_vface_lmdb(dataset_size="not-a-size", local_dir=tmp_path)


def test_download_hsface_lmdb_rejects_unknown_size(tmp_path) -> None:
    with pytest.raises(ValueError, match="dataset_size"):
        download_hsface_lmdb(dataset_size="not-a-size", local_dir=tmp_path)


def test_vface_dataset_mapping_has_released_sizes() -> None:
    assert VFACE_DATASETS["10k"] == "VFaces/vface10k.lmdb"
    assert VFACE_DATASETS["20k"] == "VFaces/vface20k.lmdb"
    assert VFACE_DATASETS["100k"] == "VFaces/vface100k.lmdb"


def test_hsface_dataset_mapping_has_published_sizes() -> None:
    assert HSFACE_DATASETS["10k"] == "HSFaces/hsface10k.lmdb"
    assert HSFACE_DATASETS["20k"] == "HSFaces/hsface20k.lmdb"


def test_rank_identity_samples_prefers_centroid_similarity() -> None:
    image_paths = tuple(Path(f"{index:03d}.jpg") for index in range(4))
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.97, 0.03],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    qualities = np.ones(4, dtype=np.float32)

    ranked = rank_identity_samples(
        identity="0001",
        image_paths=image_paths,
        embeddings=embeddings,
        qualities=qualities,
        keep_per_identity=2,
        anchor_count=2,
        quality_weight=0.0,
    )

    selected = {sample.source_path.name for sample in ranked if sample.selected}
    assert "003.jpg" not in selected
    assert len(selected) == 2
    assert ranked[-1].source_path.name == "003.jpg"


def test_rank_identity_samples_uses_quality_as_tiebreaker() -> None:
    image_paths = tuple(Path(f"{index:03d}.jpg") for index in range(3))
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    qualities = np.array([0.1, 10.0, 1.0], dtype=np.float32)

    ranked = rank_identity_samples(
        identity="0001",
        image_paths=image_paths,
        embeddings=embeddings,
        qualities=qualities,
        keep_per_identity=1,
        quality_weight=0.5,
    )

    selected = [sample.source_path.name for sample in ranked if sample.selected]
    assert selected == ["001.jpg"]


def test_summarize_center_scout_samples_accepts_stable_center() -> None:
    samples = (
        AdaFaceSample("0007", Path("0007/000.jpg"), 0.99, 10.0, 0.99, True),
        AdaFaceSample("0007", Path("0007/001.jpg"), 0.98, 9.0, 0.98, True),
        AdaFaceSample("0007", Path("0007/002.jpg"), 0.96, 8.0, 0.96, False),
        AdaFaceSample("0007", Path("0007/003.jpg"), 0.91, 7.0, 0.91, False),
    )

    summaries = summarize_center_scout_samples(
        samples,
        thresholds=CenterScoutThresholds(
            min_similarity=0.95,
            min_yield_rate=0.50,
            min_selected=2,
            min_images=4,
        ),
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.center_index == 7
    assert summary.eligible_count == 3
    assert summary.selected_count == 2
    assert summary.accepted is True
    assert summary.reason == "accepted"


def test_summarize_center_scout_samples_rejects_low_yield() -> None:
    samples = (
        AdaFaceSample("0008", Path("0008/000.jpg"), 0.99, 10.0, 0.99, True),
        AdaFaceSample("0008", Path("0008/001.jpg"), 0.90, 9.0, 0.90, False),
        AdaFaceSample("0008", Path("0008/002.jpg"), 0.89, 8.0, 0.89, False),
        AdaFaceSample("0008", Path("0008/003.jpg"), 0.88, 7.0, 0.88, False),
    )

    summary = summarize_center_scout_samples(
        samples,
        thresholds=CenterScoutThresholds(
            min_similarity=0.95,
            min_yield_rate=0.50,
            min_selected=1,
            min_images=4,
        ),
    )[0]

    assert summary.accepted is False
    assert summary.reason == "low_yield"


def test_build_center_sampling_plan_keeps_accepted_centers() -> None:
    center_features = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [8.0, 0.0],
        ],
        dtype=np.float32,
    )
    summaries = (
        center_summary(center_index=0, accepted=True),
        center_summary(center_index=1, accepted=False),
        center_summary(center_index=2, accepted=True),
    )

    rows = build_center_sampling_plan(
        center_features=center_features,
        scout_summaries=summaries,
        examples=6,
        retry_examples=2,
    )

    assert [row.center_index for row in rows] == [0, 2]
    assert all(row.accepted for row in rows)
    assert all(row.examples == 6 for row in rows)


def test_build_center_sampling_plan_can_include_retries() -> None:
    center_features = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    summaries = (
        center_summary(center_index=0, accepted=True),
        center_summary(center_index=1, accepted=False),
    )

    rows = build_center_sampling_plan(
        center_features=center_features,
        scout_summaries=summaries,
        examples=6,
        retry_examples=2,
        include_rejected=True,
    )

    retry = next(row for row in rows if row.center_index == 1)
    assert retry.accepted is False
    assert retry.examples == 2
    assert retry.schedule.name == "retry"


def test_export_center_feature_subsets_writes_mapping_and_commands(tmp_path) -> None:
    center_features = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [8.0, 0.0],
            [16.0, 0.0],
        ],
        dtype=np.float32,
    )
    rows = build_center_sampling_plan(
        center_features=center_features,
        examples=3,
        low_norm_quantile=0.25,
        high_norm_quantile=0.75,
    )

    subsets = export_center_feature_subsets(
        center_features=center_features,
        plan_rows=rows,
        output_dir=tmp_path,
        prefix="centers",
    )
    commands_path = write_generation_commands(
        subsets=subsets,
        output_path=tmp_path / "generate_commands.sh",
        run_name_prefix="scaled",
        module_python=".venv/bin/python",
        batch_size=16,
    )

    assert subsets
    assert all(subset.feature_path.exists() for subset in subsets)
    assert all(subset.mapping_path.exists() for subset in subsets)
    assert sum(np.load(subset.feature_path).shape[0] for subset in subsets) == len(rows)
    assert "generate-from-center-features" in commands_path.read_text(encoding="utf-8")


def center_summary(*, center_index: int, accepted: bool) -> CenterScoutSummary:
    return CenterScoutSummary(
        identity=f"{center_index:04d}",
        center_index=center_index,
        image_count=10,
        eligible_count=8 if accepted else 1,
        selected_count=6 if accepted else 1,
        yield_rate=0.8 if accepted else 0.1,
        similarity_min=0.92,
        similarity_p10=0.95,
        similarity_median=0.98,
        similarity_mean=0.97,
        selected_similarity_min=0.96 if accepted else 0.91,
        selected_similarity_median=0.98 if accepted else 0.91,
        quality_p10=8.0,
        quality_median=10.0,
        accepted=accepted,
        reason="accepted" if accepted else "low_yield",
    )


def test_select_diverse_indices_penalizes_near_duplicates() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.999, 0.001],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = np.array([1.0, 0.99, 0.95], dtype=np.float32)

    selected = select_diverse_indices(
        embeddings=embeddings,
        scores=scores,
        eligible=np.array([0, 1, 2]),
        keep=2,
        diversity_weight=0.2,
    )

    assert selected == [0, 2]


def test_augment_identity_dataset_preserves_identity_folders(tmp_path) -> None:
    source_dir = tmp_path / "source"
    identity_dir = source_dir / "0001"
    identity_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), color=(120, 100, 80)).save(identity_dir / "000.jpg")
    Image.new("RGB", (16, 16), color=(80, 110, 130)).save(identity_dir / "001.jpg")

    result = augment_identity_dataset(
        source_dir=source_dir,
        output_dir=tmp_path / "augmented",
        variants_per_image=2,
        include_originals=True,
        seed=7,
    )

    output_images = sorted((tmp_path / "augmented" / "0001").glob("*.jpg"))
    assert result.identities == 1
    assert result.source_images == 2
    assert result.output_images == 6
    assert [path.name for path in output_images] == [
        "000.jpg",
        "001.jpg",
        "002.jpg",
        "003.jpg",
        "004.jpg",
        "005.jpg",
    ]


def test_augment_pose_identity_dataset_preserves_identity_folders(tmp_path) -> None:
    source_dir = tmp_path / "source"
    identity_dir = source_dir / "0002"
    identity_dir.mkdir(parents=True)
    Image.new("RGB", (24, 24), color=(120, 100, 80)).save(identity_dir / "000.jpg")
    Image.new("RGB", (24, 24), color=(80, 110, 130)).save(identity_dir / "001.jpg")

    result = augment_pose_identity_dataset(
        source_dir=source_dir,
        output_dir=tmp_path / "pose_augmented",
        variants_per_image=1,
        include_originals=True,
        seed=11,
    )

    output_images = sorted((tmp_path / "pose_augmented" / "0002").glob("*.jpg"))
    assert result.identities == 1
    assert result.source_images == 2
    assert result.output_images == 4
    assert [path.name for path in output_images] == [
        "000.jpg",
        "001.jpg",
        "002.jpg",
        "003.jpg",
    ]
