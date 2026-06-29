"""Property and unit tests for the pure (CPU-only) synthetic-face helpers."""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from PIL import Image

from llm_spark_exp.synthetic_faces._common import (
    collect_identity_images,
    iter_identity_dirs,
    list_identity_images,
    sample_range,
    save_jpeg,
)
from llm_spark_exp.synthetic_faces.augment.photometric import (
    PhotometricAugmentConfig,
    augment_image,
)
from llm_spark_exp.synthetic_faces.augment.pose import (
    PoseAugmentConfig,
    augment_pose_image,
    estimate_fill_color,
    perspective_coefficients,
)
from llm_spark_exp.synthetic_faces.planning.center_sampling import (
    CenterScoutSummary,
    CenterScoutThresholds,
    classify_norm_tier,
    decide_center_acceptance,
    format_float,
    load_center_scout_summary_csv,
    parse_center_index,
    percentile,
    write_center_scout_summary_csv,
)
from llm_spark_exp.synthetic_faces.quality.identity_consistency import normalize_vector, zscore

# --------------------------------------------------------------------------- #
# _common helpers
# --------------------------------------------------------------------------- #


def _write_image(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def test_collect_identity_images_groups_and_skips_empty(tmp_path: Path) -> None:
    _write_image(tmp_path / "0001" / "a.jpg")
    _write_image(tmp_path / "0001" / "b.png")
    _write_image(tmp_path / "0002" / "c.jpeg")
    (tmp_path / "0003_empty").mkdir()
    (tmp_path / "0004_no_images").mkdir()
    (tmp_path / "0004_no_images" / "notes.txt").write_text("x", encoding="utf-8")

    collected = collect_identity_images(tmp_path)

    assert set(collected) == {"0001", "0002"}
    assert len(collected["0001"]) == 2
    # results are sorted within an identity
    assert list(collected["0001"]) == sorted(collected["0001"])


def test_iter_identity_dirs_is_sorted(tmp_path: Path) -> None:
    for name in ("c", "a", "b"):
        (tmp_path / name).mkdir()
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")

    dirs = iter_identity_dirs(tmp_path)

    assert [d.name for d in dirs] == ["a", "b", "c"]


def test_list_identity_images_filters_by_suffix(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.JPG")  # case-insensitive suffix
    _write_image(tmp_path / "b.webp")
    (tmp_path / "c.txt").write_text("x", encoding="utf-8")

    images = list_identity_images(tmp_path)

    assert [p.name for p in images] == ["a.JPG", "b.webp"]


def test_save_jpeg_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "x.jpg"
    save_jpeg(Image.new("RGB", (16, 16), color=(100, 110, 120)), out, quality=90)

    assert out.exists()
    with Image.open(out) as reread:
        assert reread.size == (16, 16)
        assert reread.mode == "RGB"


@given(
    lo=st.floats(min_value=-1e3, max_value=1e3),
    width=st.floats(min_value=0.0, max_value=1e3),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_sample_range_stays_within_bounds(lo: float, width: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    value = sample_range(rng, (lo, lo + width))
    assert lo <= value <= lo + width


# --------------------------------------------------------------------------- #
# identity_consistency numeric helpers
# --------------------------------------------------------------------------- #


@given(
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=64,
    )
)
def test_normalize_vector_is_unit_norm(values: list[float]) -> None:
    vector = np.asarray(values, dtype=np.float64)
    assume(np.linalg.norm(vector) > 1e-6)

    normalized = normalize_vector(vector)

    assert np.isclose(np.linalg.norm(normalized), 1.0, atol=1e-9)


def test_normalize_vector_zero_is_unchanged() -> None:
    zero = np.zeros(8, dtype=np.float64)
    assert np.array_equal(normalize_vector(zero), zero)


@given(
    st.lists(
        st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=64,
    )
)
def test_zscore_has_zero_mean_unit_std(values: list[float]) -> None:
    array = np.asarray(values, dtype=np.float64)
    assume(array.std() > 1e-6)

    standardized = zscore(array)

    assert np.isclose(standardized.mean(), 0.0, atol=1e-6)
    assert np.isclose(standardized.std(), 1.0, atol=1e-6)


def test_zscore_constant_input_is_zero() -> None:
    array = np.full(5, 3.0)
    assert np.array_equal(zscore(array), np.zeros(5))


# --------------------------------------------------------------------------- #
# center_sampling helpers
# --------------------------------------------------------------------------- #


def test_classify_norm_tier_boundaries() -> None:
    assert classify_norm_tier(1.0, low_cutoff=1.0, high_cutoff=3.0) == "low"
    assert classify_norm_tier(2.0, low_cutoff=1.0, high_cutoff=3.0) == "mid"
    assert classify_norm_tier(3.0, low_cutoff=1.0, high_cutoff=3.0) == "high"


@pytest.mark.parametrize(
    ("image_count", "eligible_count", "yield_rate", "expected"),
    [
        (3, 3, 1.0, (False, "insufficient_images")),
        (4, 1, 0.25, (False, "too_few_eligible")),
        (4, 3, 0.40, (False, "low_yield")),
        (4, 4, 1.0, (True, "accepted")),
    ],
)
def test_decide_center_acceptance(image_count, eligible_count, yield_rate, expected) -> None:
    thresholds = CenterScoutThresholds(
        min_similarity=0.95, min_yield_rate=0.60, min_selected=2, min_images=4
    )
    assert (
        decide_center_acceptance(
            image_count=image_count,
            eligible_count=eligible_count,
            yield_rate=yield_rate,
            thresholds=thresholds,
        )
        == expected
    )


def test_parse_center_index() -> None:
    assert parse_center_index("0007") == 7
    assert parse_center_index("not_numeric") is None


def test_percentile_handles_empty() -> None:
    assert np.isnan(percentile(np.asarray([]), 50))
    assert percentile(np.asarray([1.0, 2.0, 3.0]), 50) == 2.0


def test_format_float_is_compact() -> None:
    assert format_float(0.1234567891) == "0.12345679"


def test_center_scout_summary_csv_roundtrip(tmp_path: Path) -> None:
    summary = CenterScoutSummary(
        identity="0007",
        center_index=7,
        image_count=10,
        eligible_count=8,
        selected_count=6,
        yield_rate=0.8,
        similarity_min=0.92,
        similarity_p10=0.95,
        similarity_median=0.98,
        similarity_mean=0.97,
        selected_similarity_min=0.96,
        selected_similarity_median=0.98,
        quality_p10=8.0,
        quality_median=10.0,
        accepted=True,
        reason="accepted",
    )
    path = tmp_path / "summary.csv"
    write_center_scout_summary_csv([summary], path)

    (restored,) = load_center_scout_summary_csv(path)

    assert restored.identity == summary.identity
    assert restored.center_index == summary.center_index
    assert restored.accepted is True
    assert restored.reason == "accepted"
    assert np.isclose(restored.yield_rate, summary.yield_rate)
    assert np.isclose(restored.selected_similarity_median, summary.selected_similarity_median)


# --------------------------------------------------------------------------- #
# pose geometry
# --------------------------------------------------------------------------- #


def test_perspective_coefficients_identity_roundtrip() -> None:
    rect = ((0.0, 0.0), (32.0, 0.0), (32.0, 24.0), (0.0, 24.0))
    coeffs = perspective_coefficients(output_points=rect, input_points=rect)
    # mapping a rectangle onto itself must yield the identity transform
    assert np.allclose(coeffs, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), atol=1e-9)


def test_estimate_fill_color_returns_valid_rgb() -> None:
    image = Image.new("RGB", (20, 20), color=(40, 80, 120))
    color = estimate_fill_color(image)
    assert color == (40, 80, 120)
    assert all(0 <= c <= 255 for c in color)


def test_augment_pose_image_preserves_size_and_mode() -> None:
    rng = np.random.default_rng(0)
    image = Image.new("RGB", (40, 32), color=(90, 100, 110))
    out = augment_pose_image(image, rng=rng, config=PoseAugmentConfig())
    assert out.size == (40, 32)
    assert out.mode == "RGB"


def test_augment_image_preserves_size_and_mode() -> None:
    rng = np.random.default_rng(0)
    image = Image.new("RGB", (40, 32), color=(90, 100, 110))
    out = augment_image(image, rng=rng, config=PhotometricAugmentConfig())
    assert out.size == (40, 32)
    assert out.mode == "RGB"
