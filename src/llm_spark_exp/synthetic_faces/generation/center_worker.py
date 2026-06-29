"""Worker for generating Vec2Face+ identities from center features."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Vec2Face+ center-feature generation")
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--center-feature", required=True, type=Path)
    parser.add_argument("--model-weights", required=True, type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--examples", default=1, type=int)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--start-end", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=18, type=int)
    parser.add_argument("--skip-feature-refinement", action="store_true")
    parser.add_argument(
        "--variation-sigmas",
        default="0.08,0.12,0.18",
        type=str,
        help="Comma-separated noise scales for intraclass variation.",
    )
    parser.add_argument(
        "--variation-weights",
        default="0.5,0.35,0.15",
        type=str,
        help="Comma-separated proportions for each variation sigma.",
    )
    return parser.parse_args()


def sample_nearby_vectors(
    base_vector: torch.Tensor,
    *,
    epsilons: tuple[float, ...],
    percentages: tuple[float, ...],
) -> torch.Tensor:
    """Sample identity-preserving neighbors by adding scaled Gaussian noise.

    Rows of ``base_vector`` are split across the noise scales (``epsilons``) by
    the given ``percentages``, perturbed, then renormalized to each row's
    original norm so identity magnitude is preserved.
    """

    if len(epsilons) != len(percentages):
        raise ValueError("variation sigmas and weights must have the same length.")
    if not np.isclose(sum(percentages), 1.0):
        raise ValueError("variation weights must sum to 1.0.")

    row, col = base_vector.shape
    norm = torch.norm(base_vector, 2, 1, True)
    diff_parts = []
    assigned_rows = 0
    for index, epsilon in enumerate(epsilons):
        part_rows = int(row * percentages[index])
        if index == len(epsilons) - 1:
            part_rows = row - assigned_rows
        assigned_rows += part_rows
        diff_parts.append(np.random.normal(0, epsilon, (part_rows, col)))
    diff = np.vstack(diff_parts)
    np.random.shuffle(diff)
    generated_samples = base_vector + torch.tensor(diff, dtype=base_vector.dtype)
    return generated_samples / torch.norm(generated_samples, 2, 1, True) * norm


def parse_float_tuple(value: str, *, name: str) -> tuple[float, ...]:
    """Parse a comma-separated string into a tuple of positive floats."""

    items = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not items:
        raise ValueError(f"{name} must contain at least one value.")
    if any(item <= 0 for item in items):
        raise ValueError(f"{name} values must be positive.")
    return items


def create_fr_model(model_path: Path, *, depth: str = "100") -> torch.nn.Module:
    """Load an IResNet face-recognition model from the Vec2Face+ checkout."""

    from models import iresnet

    model = iresnet(depth)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def save_images(images: np.ndarray, id_nums: torch.Tensor, *, root: Path, name: str) -> None:
    """Write generated images into per-identity folders with dense numbering."""

    save_root = root / name
    previous_id = None
    image_indices: dict[int, int] = {}
    for index, image in enumerate(images):
        id_num = int(id_nums[index])
        save_folder = save_root / f"{id_num:04d}"
        save_folder.mkdir(parents=True, exist_ok=True)
        if previous_id != id_num:
            previous_id = id_num
            image_indices.setdefault(id_num, len(list(save_folder.glob("*.jpg"))))
        image_index = image_indices[id_num]
        imageio.imwrite(save_folder / f"{image_index:03d}.jpg", image)
        image_indices[id_num] = image_index + 1


def main() -> None:
    """Generate identities from center features and write them under the repo output root."""

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    repo_dir = args.repo_dir.resolve()
    os.chdir(repo_dir)
    sys.path.insert(0, str(repo_dir))

    import pixel_generator.vec2face.model_vec2face as model_vec2face

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    device = torch.device(args.device)

    model = model_vec2face.vec2face_vit_base_patch16(
        mask_ratio_mu=0.15,
        mask_ratio_std=0.25,
        mask_ratio_min=0.1,
        mask_ratio_max=1.0,
        use_rep=True,
        rep_dim=512,
        rep_drop_prob=0.0,
        use_class_label=False,
    ).to(device)
    checkpoint = torch.load(args.model_weights, map_location=device)
    model.load_state_dict(checkpoint["model_vec2face"])
    model.eval()

    random_ids = torch.tensor(np.load(args.center_feature), dtype=torch.float32)
    if args.start_end is None:
        start, end = 0, len(random_ids)
    else:
        start_text, end_text = args.start_end.split(":")
        start, end = int(start_text), int(end_text)
        if end <= start:
            raise ValueError("--start-end must have an end greater than its start.")

    selected_ids = random_ids[start:end]
    class_labels = torch.arange(start, end)

    if args.skip_feature_refinement:
        updated_features = selected_ids
    else:
        scorer = create_fr_model(repo_dir / "weights" / "magface-r100-glint360k.pth").to(device)
        fr_model = create_fr_model(repo_dir / "weights" / "arcface-r100-glint360k.pth").to(device)
        updated_features = torch.empty_like(selected_ids)
        for index in tqdm(range(0, len(selected_ids), args.batch_size), desc="Refining identities"):
            im_features = selected_ids[index : index + args.batch_size].to(
                device, non_blocking=True
            )
            _, im_features = model.gen_image(im_features, scorer, fr_model, class_rep=im_features)
            updated_features[index : index + args.batch_size] = im_features.cpu()

    expanded_ids = torch.repeat_interleave(updated_features, args.examples, dim=0).to(torch.float32)
    expanded_labels = torch.repeat_interleave(class_labels, args.examples, dim=0)
    samples = (
        sample_nearby_vectors(
            expanded_ids,
            epsilons=parse_float_tuple(args.variation_sigmas, name="variation sigmas"),
            percentages=parse_float_tuple(args.variation_weights, name="variation weights"),
        )
        .to(torch.float32)
        .to(device, non_blocking=True)
    )

    output_root = repo_dir / "generated_images"
    for index in tqdm(range(0, len(samples), args.batch_size), desc="Generating images"):
        im_features = samples[index : index + args.batch_size]
        _, _, image, *_ = model(im_features)
        images = (
            ((image.permute(0, 2, 3, 1).detach().cpu().numpy() + 1) / 2 * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )
        save_images(
            images,
            expanded_labels[index : index + args.batch_size],
            root=output_root,
            name=args.name,
        )

    print(output_root / args.name)


if __name__ == "__main__":
    main()
