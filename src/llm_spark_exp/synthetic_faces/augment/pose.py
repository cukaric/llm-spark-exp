"""Mild geometric augmentation for aligned synthetic face crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from llm_spark_exp.synthetic_faces._common import (
    iter_identity_dirs,
    list_identity_images,
    sample_range,
    save_jpeg,
)


@dataclass(frozen=True)
class PoseAugmentResult:
    """Summary of a pose augmentation run."""

    source_dir: Path
    output_dir: Path
    identities: int
    source_images: int
    output_images: int


@dataclass(frozen=True)
class PoseAugmentConfig:
    """Ranges for mild crop-safe geometric variation."""

    roll_degrees: tuple[float, float] = (-7.0, 7.0)
    yaw_strength: tuple[float, float] = (-0.12, 0.12)
    shear_strength: tuple[float, float] = (-0.035, 0.035)
    scale: tuple[float, float] = (0.96, 1.05)
    translate_x: tuple[float, float] = (-0.035, 0.035)
    translate_y: tuple[float, float] = (-0.025, 0.035)
    jpeg_quality: tuple[int, int] = (90, 98)


def augment_pose_identity_dataset(
    *,
    source_dir: Path,
    output_dir: Path,
    variants_per_image: int = 1,
    include_originals: bool = True,
    seed: int = 23,
    config: PoseAugmentConfig | None = None,
) -> PoseAugmentResult:
    """Create mild geometric variants of one-folder-per-identity images."""

    if variants_per_image < 1:
        raise ValueError("variants_per_image must be at least 1.")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    config = config or PoseAugmentConfig()
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    identities = 0
    source_images = 0
    output_images = 0
    for identity_dir in iter_identity_dirs(source_dir):
        images = list_identity_images(identity_dir)
        if not images:
            continue
        identities += 1
        destination_identity_dir = output_dir / identity_dir.name
        destination_identity_dir.mkdir(parents=True, exist_ok=True)
        next_index = 0
        for image_path in images:
            source_images += 1
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                if include_originals:
                    save_jpeg(rgb, destination_identity_dir / f"{next_index:03d}.jpg", quality=98)
                    next_index += 1
                    output_images += 1
                for _ in range(variants_per_image):
                    augmented = augment_pose_image(rgb, rng=rng, config=config)
                    quality = int(rng.integers(config.jpeg_quality[0], config.jpeg_quality[1] + 1))
                    save_jpeg(
                        augmented,
                        destination_identity_dir / f"{next_index:03d}.jpg",
                        quality=quality,
                    )
                    next_index += 1
                    output_images += 1

    return PoseAugmentResult(
        source_dir=source_dir,
        output_dir=output_dir,
        identities=identities,
        source_images=source_images,
        output_images=output_images,
    )


def augment_pose_image(
    image: Image.Image,
    *,
    rng: np.random.Generator,
    config: PoseAugmentConfig,
) -> Image.Image:
    """Apply one mild pose-like geometric augmentation."""

    fill = estimate_fill_color(image)
    transformed = apply_affine_pose(image, rng=rng, config=config, fill=fill)
    yaw = sample_range(rng, config.yaw_strength)
    if abs(yaw) < 0.015:
        return transformed
    return apply_yaw_perspective(transformed, yaw=yaw, fill=fill)


def apply_affine_pose(
    image: Image.Image,
    *,
    rng: np.random.Generator,
    config: PoseAugmentConfig,
    fill: tuple[int, int, int],
) -> Image.Image:
    width, height = image.size
    angle = np.deg2rad(sample_range(rng, config.roll_degrees))
    shear = sample_range(rng, config.shear_strength)
    scale = sample_range(rng, config.scale)
    translate_x = sample_range(rng, config.translate_x) * width
    translate_y = sample_range(rng, config.translate_y) * height

    cos_a = np.cos(angle) * scale
    sin_a = np.sin(angle) * scale
    center_x = width / 2.0
    center_y = height / 2.0
    forward = np.array(
        [
            [cos_a, -sin_a + shear, center_x + translate_x],
            [sin_a, cos_a, center_y + translate_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    to_origin = np.array(
        [[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    matrix = forward @ to_origin
    inverse = np.linalg.inv(matrix)
    coeffs = tuple(inverse[:2, :].reshape(-1))
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=fill,
    )


def apply_yaw_perspective(
    image: Image.Image,
    *,
    yaw: float,
    fill: tuple[int, int, int],
) -> Image.Image:
    width, height = float(image.size[0]), float(image.size[1])
    inset = abs(yaw) * width * 0.16
    vertical_shift = abs(yaw) * height * 0.035
    if yaw > 0:
        destination = (
            (0.0, vertical_shift),
            (width - inset, 0.0),
            (width - inset, height),
            (0.0, height - vertical_shift),
        )
    else:
        destination = (
            (inset, 0.0),
            (width, vertical_shift),
            (width, height - vertical_shift),
            (inset, height),
        )
    source = ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
    coeffs = perspective_coefficients(output_points=destination, input_points=source)
    return image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=fill,
    )


def perspective_coefficients(
    output_points: tuple[tuple[float, float], ...],
    input_points: tuple[tuple[float, float], ...],
) -> tuple[float, ...]:
    """Solve PIL perspective coefficients mapping output pixels back to input pixels.

    ``Image.Transform.PERSPECTIVE`` samples the input image for every output
    pixel, so the coefficients map output (destination-image) coordinates to
    input (source-image) coordinates. Pass the warped destination quad as
    ``output_points`` and the original rectangle as ``input_points``.
    """

    matrix = []
    values = []
    for (x, y), (u, v) in zip(output_points, input_points, strict=True):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.extend([u, v])
    return tuple(np.linalg.solve(np.asarray(matrix), np.asarray(values)))


def estimate_fill_color(image: Image.Image) -> tuple[int, int, int]:
    array = np.asarray(image, dtype=np.uint8)
    border = np.concatenate(
        [
            array[:3, :, :].reshape(-1, 3),
            array[-3:, :, :].reshape(-1, 3),
            array[:, :3, :].reshape(-1, 3),
            array[:, -3:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    color = np.median(border, axis=0)
    return (int(color[0]), int(color[1]), int(color[2]))
