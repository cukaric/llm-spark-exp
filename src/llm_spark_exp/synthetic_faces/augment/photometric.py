"""Photometric augmentation for aligned synthetic face crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from llm_spark_exp.synthetic_faces._common import (
    iter_identity_dirs,
    list_identity_images,
    sample_range,
    save_jpeg,
)


@dataclass(frozen=True)
class PhotometricAugmentResult:
    """Summary of a photometric augmentation run."""

    source_dir: Path
    output_dir: Path
    identities: int
    source_images: int
    output_images: int


@dataclass(frozen=True)
class PhotometricAugmentConfig:
    """Ranges for mild FR-safe photometric variation."""

    brightness: tuple[float, float] = (0.82, 1.18)
    contrast: tuple[float, float] = (0.82, 1.18)
    saturation: tuple[float, float] = (0.85, 1.15)
    gamma: tuple[float, float] = (0.85, 1.18)
    color_temperature: tuple[float, float] = (-0.08, 0.08)
    shadow_strength: tuple[float, float] = (0.0, 0.18)
    blur_probability: float = 0.12
    blur_radius: tuple[float, float] = (0.2, 0.65)
    noise_std: tuple[float, float] = (0.0, 3.0)
    jpeg_quality: tuple[int, int] = (88, 98)


def augment_identity_dataset(
    *,
    source_dir: Path,
    output_dir: Path,
    variants_per_image: int = 1,
    include_originals: bool = True,
    seed: int = 18,
    config: PhotometricAugmentConfig | None = None,
) -> PhotometricAugmentResult:
    """Create photometrically augmented copies of one-folder-per-identity images."""

    if variants_per_image < 1:
        raise ValueError("variants_per_image must be at least 1.")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    config = config or PhotometricAugmentConfig()
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
                    augmented = augment_image(rgb, rng=rng, config=config)
                    quality = int(rng.integers(config.jpeg_quality[0], config.jpeg_quality[1] + 1))
                    save_jpeg(
                        augmented,
                        destination_identity_dir / f"{next_index:03d}.jpg",
                        quality=quality,
                    )
                    next_index += 1
                    output_images += 1

    return PhotometricAugmentResult(
        source_dir=source_dir,
        output_dir=output_dir,
        identities=identities,
        source_images=source_images,
        output_images=output_images,
    )


def augment_image(
    image: Image.Image,
    *,
    rng: np.random.Generator,
    config: PhotometricAugmentConfig,
) -> Image.Image:
    """Apply one mild random photometric augmentation."""

    augmented = image.copy()
    augmented = ImageEnhance.Brightness(augmented).enhance(sample_range(rng, config.brightness))
    augmented = ImageEnhance.Contrast(augmented).enhance(sample_range(rng, config.contrast))
    augmented = ImageEnhance.Color(augmented).enhance(sample_range(rng, config.saturation))
    augmented = apply_gamma(augmented, gamma=sample_range(rng, config.gamma))
    augmented = apply_color_temperature(
        augmented,
        shift=sample_range(rng, config.color_temperature),
    )
    augmented = apply_soft_shadow(
        augmented,
        strength=sample_range(rng, config.shadow_strength),
        rng=rng,
    )
    if rng.random() < config.blur_probability:
        augmented = augmented.filter(
            ImageFilter.GaussianBlur(radius=sample_range(rng, config.blur_radius))
        )
    return apply_noise(augmented, std=sample_range(rng, config.noise_std), rng=rng)


def apply_gamma(image: Image.Image, *, gamma: float) -> Image.Image:
    array = np.asarray(image, dtype=np.float32) / 255.0
    corrected = np.power(np.clip(array, 0.0, 1.0), gamma)
    return Image.fromarray((corrected * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")


def apply_color_temperature(image: Image.Image, *, shift: float) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    array[..., 0] *= 1.0 + shift
    array[..., 2] *= 1.0 - shift
    return Image.fromarray(array.clip(0, 255).astype(np.uint8), mode="RGB")


def apply_soft_shadow(
    image: Image.Image,
    *,
    strength: float,
    rng: np.random.Generator,
) -> Image.Image:
    if strength <= 0:
        return image
    width, height = image.size
    x = np.linspace(-1, 1, width, dtype=np.float32)
    y = np.linspace(-1, 1, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)
    angle = sample_range(rng, (0, 2 * np.pi))
    gradient = (np.cos(angle) * grid_x + np.sin(angle) * grid_y + 1.0) / 2.0
    mask = 1.0 - strength * gradient[..., None]
    array = np.asarray(image, dtype=np.float32) * mask
    return Image.fromarray(array.clip(0, 255).astype(np.uint8), mode="RGB")


def apply_noise(
    image: Image.Image,
    *,
    std: float,
    rng: np.random.Generator,
) -> Image.Image:
    if std <= 0:
        return image
    array = np.asarray(image, dtype=np.float32)
    noise = rng.normal(0.0, std, size=array.shape)
    return Image.fromarray((array + noise).clip(0, 255).astype(np.uint8), mode="RGB")
