"""Extract a compact Vec2Face+ pose landmark reference bank from an LMDB."""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from pathlib import Path

import lmdb
import msgpack
import numpy as np
from PIL import Image


def load_pose_estimator(weights_path: Path, *, device: str):
    """Load the locally installed 6DRepNet model for yaw/pitch/roll estimates."""

    import sixdrepnet
    import torch
    from torchvision import transforms

    package_dir = Path(sixdrepnet.__file__).parent
    if str(package_dir) not in sys.path:
        sys.path.append(str(package_dir))

    import utils as sixd_utils
    from sixdrepnet.model import SixDRepNet

    model = SixDRepNet(
        backbone_name="RepVGG-B1g2",
        backbone_file="",
        deploy=True,
        pretrained=False,
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return model, transform, sixd_utils


def landmark_features(image: Image.Image) -> np.ndarray:
    """Summarize a sparse RGB landmark map for diversity selection."""

    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    features = []
    for channel in range(3):
        weights = array[:, :, channel]
        mask = weights > 20.0
        if not np.any(mask):
            features.extend([0.0, 0.0, 0.0, 0.0])
            continue
        ys, xs = np.nonzero(mask)
        values = weights[mask]
        total = float(values.sum())
        center_x = float((xs * values).sum() / total)
        center_y = float((ys * values).sum() / total)
        spread_x = float(np.sqrt((((xs - center_x) ** 2) * values).sum() / total))
        spread_y = float(np.sqrt((((ys - center_y) ** 2) * values).sum() / total))
        features.extend([center_x, center_y, spread_x, spread_y])
    return np.asarray(features, dtype=np.float32)


def is_usable_landmark(features: np.ndarray, *, margin: float) -> bool:
    """Reject maps where colored landmarks are mostly missing or off-crop."""

    for offset in range(0, len(features), 4):
        center_x, center_y, spread_x, spread_y = features[offset : offset + 4]
        if center_x == 0.0 and center_y == 0.0:
            return False
        if not (margin <= center_x <= 112.0 - margin):
            return False
        if not (margin <= center_y <= 112.0 - margin):
            return False
        if spread_x < 0.5 or spread_y < 0.5:
            return False
    return True


def farthest_first(features: np.ndarray, count: int) -> list[int]:
    """Pick diverse examples with deterministic farthest-first traversal."""

    if count >= len(features):
        return list(range(len(features)))

    normalized = features.copy()
    scale = normalized.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (normalized - normalized.mean(axis=0)) / scale

    selected = [int(np.argmax(np.linalg.norm(normalized - normalized.mean(axis=0), axis=1)))]
    min_distances = np.linalg.norm(normalized - normalized[selected[0]], axis=1)
    while len(selected) < count:
        index = int(np.argmax(min_distances))
        selected.append(index)
        distances = np.linalg.norm(normalized - normalized[index], axis=1)
        min_distances = np.minimum(min_distances, distances)
    return selected


def estimate_pose_batch(
    images: list[Image.Image],
    *,
    model,
    transform,
    sixd_utils,
    device: str,
) -> np.ndarray:
    """Return pitch/yaw/roll estimates in degrees for aligned face crops."""

    import torch

    tensors = torch.stack([transform(image.convert("RGB")) for image in images]).to(device)
    with torch.no_grad():
        rotations = model(tensors)
        eulers = sixd_utils.compute_euler_angles_from_rotation_matrices(rotations)
    return (eulers.detach().cpu().numpy() * 180.0 / np.pi).astype(np.float32)


def pose_selection_features(
    pose: np.ndarray,
    landmark_feature: np.ndarray,
    *,
    pose_weight: float,
    landmark_weight: float,
) -> np.ndarray:
    """Build normalized features for diverse pose-aware farthest-first selection."""

    pitch, yaw, roll = pose
    pose_features = np.asarray([yaw / 45.0, pitch / 30.0, roll / 20.0], dtype=np.float32)

    channel_xy = landmark_feature.reshape(3, 4)[:, :2]
    center = channel_xy.mean(axis=0, keepdims=True)
    relative_xy = ((channel_xy - center) / 56.0).reshape(-1)
    spread = (landmark_feature.reshape(3, 4)[:, 2:] / 28.0).reshape(-1)
    landmark_feature_vector = np.concatenate([relative_xy, spread]).astype(np.float32)

    return np.concatenate(
        [
            pose_features * pose_weight,
            landmark_feature_vector * landmark_weight,
        ]
    )


def reorder_diverse(candidates: list[dict], *, count: int) -> list[dict]:
    features = np.stack([candidate["selection_feature"] for candidate in candidates])
    return [candidates[index] for index in farthest_first(features, count)]


def select_pose_balanced(candidates: list[dict], *, count: int) -> list[dict]:
    """Select a yaw/pitch balanced set, then order it for diverse prefixes."""

    yaw_edges = np.asarray([-90, -45, -30, -18, -6, 6, 18, 30, 45, 90], dtype=np.float32)
    pitch_edges = np.asarray([-90, -18, -6, 6, 18, 90], dtype=np.float32)
    cell_count = (len(yaw_edges) - 1) * (len(pitch_edges) - 1)
    per_cell = max(1, count // cell_count)

    by_cell: dict[tuple[int, int], list[dict]] = {}
    for candidate in candidates:
        pitch, yaw, _ = candidate["pose"]
        yaw_bin = int(
            np.clip(np.searchsorted(yaw_edges, yaw, side="right") - 1, 0, len(yaw_edges) - 2)
        )
        pitch_bin = int(
            np.clip(np.searchsorted(pitch_edges, pitch, side="right") - 1, 0, len(pitch_edges) - 2)
        )
        by_cell.setdefault((yaw_bin, pitch_bin), []).append(candidate)

    selected = []
    selected_indices = set()
    for cell in sorted(by_cell):
        cell_candidates = by_cell[cell]
        take = min(per_cell, len(cell_candidates), count - len(selected))
        if take <= 0:
            continue
        ordered = reorder_diverse(cell_candidates, count=take)
        selected.extend(ordered)
        selected_indices.update(candidate["lmdb_index"] for candidate in ordered)

    if len(selected) < count:
        remaining = [
            candidate for candidate in candidates if candidate["lmdb_index"] not in selected_indices
        ]
        selected.extend(reorder_diverse(remaining, count=count - len(selected)))

    return reorder_diverse(selected, count=min(count, len(selected)))


def extract_bank(
    *,
    lmdb_path: Path,
    repo_dir: Path,
    name: str,
    sample_size: int,
    count: int,
    seed: int,
    margin: float,
    pose_weights: Path | None = None,
    device: str = "cuda",
    pose_batch_size: int = 256,
    pose_weight: float = 2.5,
    landmark_weight: float = 1.0,
    max_abs_yaw: float = 85.0,
    max_abs_pitch: float = 60.0,
    max_abs_roll: float = 45.0,
) -> Path:
    if count < 1:
        raise ValueError("count must be at least 1.")
    if sample_size < count:
        raise ValueError("sample-size must be greater than or equal to count.")
    if not lmdb_path.exists():
        raise FileNotFoundError(f"LMDB does not exist: {lmdb_path}")

    env = lmdb.open(
        str(lmdb_path),
        subdir=lmdb_path.is_dir(),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    rng = random.Random(seed)
    pose_model = pose_transform = sixd_utils = None
    if pose_weights is not None:
        pose_model, pose_transform, sixd_utils = load_pose_estimator(
            pose_weights,
            device=device,
        )

    with env.begin(write=False) as txn:
        keys = msgpack.loads(txn.get(b"__keys__"))
        indices = rng.sample(range(len(keys)), min(sample_size, len(keys)))
        candidates = []
        features = []
        pose_images = []
        pending = []
        for index in indices:
            unpacked = msgpack.loads(txn.get(keys[index]))
            face_image = Image.open(io.BytesIO(unpacked[0])).convert("RGB")
            landmark_image = Image.open(io.BytesIO(unpacked[3])).convert("RGB")
            feature = landmark_features(landmark_image)
            if not is_usable_landmark(feature, margin=margin):
                continue
            if pose_model is None:
                candidates.append(
                    {
                        "lmdb_index": index,
                        "landmark": landmark_image.copy(),
                        "pose": np.zeros(3, dtype=np.float32),
                        "landmark_feature": feature,
                        "selection_feature": feature,
                    }
                )
                features.append(feature)
                continue

            pose_images.append(face_image)
            pending.append((index, landmark_image.copy(), feature))
            if len(pose_images) >= pose_batch_size:
                poses = estimate_pose_batch(
                    pose_images,
                    model=pose_model,
                    transform=pose_transform,
                    sixd_utils=sixd_utils,
                    device=device,
                )
                for (lmdb_index, image, landmark_feature), pose in zip(
                    pending, poses, strict=False
                ):
                    pitch, yaw, roll = pose
                    if (
                        abs(float(yaw)) > max_abs_yaw
                        or abs(float(pitch)) > max_abs_pitch
                        or abs(float(roll)) > max_abs_roll
                    ):
                        continue
                    candidates.append(
                        {
                            "lmdb_index": lmdb_index,
                            "landmark": image,
                            "pose": pose,
                            "landmark_feature": landmark_feature,
                            "selection_feature": pose_selection_features(
                                pose,
                                landmark_feature,
                                pose_weight=pose_weight,
                                landmark_weight=landmark_weight,
                            ),
                        }
                    )
                    features.append(candidates[-1]["selection_feature"])
                pose_images = []
                pending = []

        if pose_images:
            poses = estimate_pose_batch(
                pose_images,
                model=pose_model,
                transform=pose_transform,
                sixd_utils=sixd_utils,
                device=device,
            )
            for (lmdb_index, image, landmark_feature), pose in zip(pending, poses, strict=False):
                pitch, yaw, roll = pose
                if (
                    abs(float(yaw)) > max_abs_yaw
                    or abs(float(pitch)) > max_abs_pitch
                    or abs(float(roll)) > max_abs_roll
                ):
                    continue
                candidates.append(
                    {
                        "lmdb_index": lmdb_index,
                        "landmark": image,
                        "pose": pose,
                        "landmark_feature": landmark_feature,
                        "selection_feature": pose_selection_features(
                            pose,
                            landmark_feature,
                            pose_weight=pose_weight,
                            landmark_weight=landmark_weight,
                        ),
                    }
                )
                features.append(candidates[-1]["selection_feature"])

    if len(candidates) < count:
        raise RuntimeError(
            f"Only found {len(candidates)} usable landmarks from {len(indices)} sampled entries. "
            "Increase --sample-size or lower --margin."
        )

    if pose_model is None:
        selected = [candidates[index] for index in farthest_first(np.stack(features), count)]
    else:
        selected = select_pose_balanced(candidates, count=count)
    output_dir = repo_dir / "landmarks" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    list_file = repo_dir / "landmarks" / f"{name}.txt"
    manifest_file = repo_dir / "landmarks" / f"{name}_manifest.csv"

    paths = []
    for rank, candidate in enumerate(selected):
        lmdb_index = candidate["lmdb_index"]
        image = candidate["landmark"]
        path = output_dir / f"{rank:03d}_lmdb_{lmdb_index:07d}.jpg"
        image.save(path, format="JPEG", quality=98)
        paths.append(path)

    with list_file.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"./{path.relative_to(repo_dir).as_posix()}\n")

    with manifest_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "lmdb_index", "pitch", "yaw", "roll", "path"])
        for rank, (candidate, path) in enumerate(zip(selected, paths, strict=False)):
            pitch, yaw, roll = candidate["pose"]
            writer.writerow(
                [
                    rank,
                    candidate["lmdb_index"],
                    f"{pitch:.4f}",
                    f"{yaw:.4f}",
                    f"{roll:.4f}",
                    f"./{path.relative_to(repo_dir).as_posix()}",
                ]
            )

    return list_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lmdb-path",
        type=Path,
        default=Path("models/vec2face_plus/repo/WebFace4M/WebFace4M_with_landmarks.lmdb"),
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("models/vec2face_plus/repo"),
    )
    parser.add_argument("--name", default="webface_pose_bank_32")
    parser.add_argument("--sample-size", type=int, default=4096)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument(
        "--margin",
        type=float,
        default=14.0,
        help="Reject landmarks whose color-channel centers are this close to an image edge.",
    )
    parser.add_argument(
        "--pose-weights",
        type=Path,
        default=None,
        help="Optional 6DRepNet checkpoint for pose-balanced landmark selection.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pose-batch-size", type=int, default=256)
    parser.add_argument("--pose-weight", type=float, default=2.5)
    parser.add_argument("--landmark-weight", type=float, default=1.0)
    parser.add_argument("--max-abs-yaw", type=float, default=85.0)
    parser.add_argument("--max-abs-pitch", type=float, default=60.0)
    parser.add_argument("--max-abs-roll", type=float, default=45.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    list_file = extract_bank(
        lmdb_path=args.lmdb_path,
        repo_dir=args.repo_dir,
        name=args.name,
        sample_size=args.sample_size,
        count=args.count,
        seed=args.seed,
        margin=args.margin,
        pose_weights=args.pose_weights,
        device=args.device,
        pose_batch_size=args.pose_batch_size,
        pose_weight=args.pose_weight,
        landmark_weight=args.landmark_weight,
        max_abs_yaw=args.max_abs_yaw,
        max_abs_pitch=args.max_abs_pitch,
        max_abs_roll=args.max_abs_roll,
    )
    print(list_file)


if __name__ == "__main__":
    main()
