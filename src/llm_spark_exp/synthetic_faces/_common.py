"""Shared filesystem and sampling helpers for the synthetic face pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from llm_spark_exp.synthetic_faces.constants import IMAGE_SUFFIXES


def iter_identity_dirs(source_dir: Path) -> list[Path]:
    """Return immediate child directories of ``source_dir``, sorted by name."""

    return sorted(path for path in source_dir.iterdir() if path.is_dir())


def list_identity_images(identity_dir: Path) -> list[Path]:
    """Return sorted image files under one identity folder (recursive)."""

    return sorted(
        path
        for path in identity_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def collect_identity_images(source_dir: Path) -> dict[str, tuple[Path, ...]]:
    """Collect images grouped by immediate child identity folder."""

    identities: dict[str, tuple[Path, ...]] = {}
    for identity_dir in iter_identity_dirs(source_dir):
        images = tuple(list_identity_images(identity_dir))
        if images:
            identities[identity_dir.name] = images
    return identities


def sample_range(rng: np.random.Generator, value_range: tuple[float, float]) -> float:
    """Sample a uniform float within an inclusive range."""

    return float(rng.uniform(value_range[0], value_range[1]))


def save_jpeg(image: Image.Image, path: Path, *, quality: int) -> None:
    """Save a PIL image as an optimized JPEG."""

    image.save(path, format="JPEG", quality=quality, optimize=True)
