"""Generate small Vec2Face+ pose landmark reference banks.

Vec2Face+ pose control expects a text file listing 112x112 RGB landmark-map
images. The upstream repo ships only two examples, so this creates a compact
synthetic bank that covers left/profile/frontal/right templates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class PoseTemplate:
    name: str
    yaw: float
    pitch: float = 0.0
    roll: float = 0.0


DEFAULT_TEMPLATES = (
    PoseTemplate("yaw_m42", yaw=-42),
    PoseTemplate("yaw_m24", yaw=-24),
    PoseTemplate("yaw_000", yaw=0),
    PoseTemplate("yaw_p24", yaw=24),
    PoseTemplate("yaw_p42", yaw=42),
    PoseTemplate("yaw_m32_pitch_up", yaw=-32, pitch=-10),
    PoseTemplate("yaw_000_pitch_up", yaw=0, pitch=-12),
    PoseTemplate("yaw_p32_pitch_up", yaw=32, pitch=-10),
    PoseTemplate("yaw_m32_pitch_down", yaw=-32, pitch=10),
    PoseTemplate("yaw_000_pitch_down", yaw=0, pitch=12),
    PoseTemplate("yaw_p32_pitch_down", yaw=32, pitch=10),
    PoseTemplate("yaw_m16_roll_m8", yaw=-16, roll=-8),
    PoseTemplate("yaw_000_roll_m8", yaw=0, roll=-8),
    PoseTemplate("yaw_p16_roll_p8", yaw=16, roll=8),
    PoseTemplate("yaw_m12_roll_p8", yaw=-12, roll=8),
    PoseTemplate("yaw_p12_roll_m8", yaw=12, roll=-8),
)


def draw_blob(
    image: Image.Image,
    *,
    center: tuple[float, float],
    radius: float,
    color: tuple[int, int, int],
) -> None:
    draw = ImageDraw.Draw(image)
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def render_template(template: PoseTemplate, *, size: int = 112) -> Image.Image:
    """Render a three-blob landmark map similar to the shipped examples."""

    yaw = max(-45.0, min(45.0, template.yaw)) / 45.0
    pitch = max(-18.0, min(18.0, template.pitch)) / 18.0
    roll = max(-12.0, min(12.0, template.roll)) / 12.0

    center_x = size / 2.0
    center_y = size / 2.0
    vertical = 4.0 * pitch

    eye = (center_x + 3.0 * yaw + 2.0 * roll, center_y - 8.0 + vertical - 2.0 * roll)
    nose = (center_x + 17.0 * yaw, center_y + 9.0 + vertical)
    mouth = (center_x - 2.0 * yaw - 2.0 * roll, center_y + 27.0 + vertical + 2.0 * roll)

    image = Image.new("RGB", (size, size), (0, 0, 0))
    draw_blob(image, center=eye, radius=4.5, color=(255, 0, 0))
    draw_blob(image, center=nose, radius=3.5, color=(0, 255, 0))
    draw_blob(image, center=mouth, radius=5.0, color=(0, 0, 255))
    return image.filter(ImageFilter.GaussianBlur(radius=1.0))


def generate_bank(*, output_dir: Path, list_file: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, template in enumerate(DEFAULT_TEMPLATES):
        path = output_dir / f"{index:02d}_{template.name}.jpg"
        render_template(template).save(path, format="JPEG", quality=98)
        paths.append(path)

    repo_dir = list_file.parent.parent
    with list_file.open("w", encoding="utf-8") as handle:
        for path in paths:
            try:
                formatted = path.relative_to(repo_dir)
            except ValueError:
                formatted = path
            handle.write(f"./{formatted.as_posix()}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("models/vec2face_plus/repo"),
        help="Local Vec2Face+ repository directory.",
    )
    parser.add_argument(
        "--name",
        default="generated_pose_bank_16",
        help="Output bank name under <repo-dir>/landmarks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    landmarks_dir = args.repo_dir / "landmarks"
    generate_bank(
        output_dir=landmarks_dir / args.name,
        list_file=landmarks_dir / f"{args.name}.txt",
    )
    print(landmarks_dir / f"{args.name}.txt")


if __name__ == "__main__":
    main()
