"""Subprocess worker for FLUX image-to-image face refinement experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_spark_exp.synthetic_faces._common import (
    iter_identity_dirs,
    list_identity_images,
    save_jpeg,
)

FLUX_MAX_SEQUENCE_LENGTH = 256  # T5 text-encoder sequence length for FLUX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FLUX image-to-image face refinement")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--cache-dir", default=Path("models/flux"), type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--strength", default=0.18, type=float)
    parser.add_argument("--num-inference-steps", default=4, type=int)
    parser.add_argument("--guidance-scale", default=0.0, type=float)
    parser.add_argument("--true-cfg-scale", default=1.0, type=float)
    parser.add_argument("--height", default=512, type=int)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--seed", default=20260611, type=int)
    parser.add_argument("--jpeg-quality", default=95, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--max-images", default=None, type=int)
    parser.add_argument("--identity-limit", default=None, type=int)
    return parser.parse_args()


def torch_dtype(name: str):
    """Map a dtype name to the matching torch dtype."""

    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def iter_images(source_dir: Path, *, identity_limit: int | None, max_images: int | None):
    """Yield (identity_name, image_path) pairs, bounded by identity/image limits."""

    yielded = 0
    identity_dirs = iter_identity_dirs(source_dir)
    if identity_limit is not None:
        identity_dirs = identity_dirs[:identity_limit]

    for identity_dir in identity_dirs:
        for image_path in list_identity_images(identity_dir):
            if max_images is not None and yielded >= max_images:
                return
            yielded += 1
            yield identity_dir.name, image_path


def main() -> None:
    """Refine every image under the source tree with FLUX image-to-image."""

    args = parse_args()

    import torch
    from diffusers import FluxImg2ImgPipeline
    from PIL import Image
    from tqdm import tqdm

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    pipe = FluxImg2ImgPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype(args.dtype),
        cache_dir=args.cache_dir,
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)

    image_items = list(
        iter_images(source_dir, identity_limit=args.identity_limit, max_images=args.max_images)
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    for identity, image_path in tqdm(image_items, desc="FLUX refine"):
        dest_identity_dir = output_dir / identity
        dest_identity_dir.mkdir(parents=True, exist_ok=True)
        output_path = dest_identity_dir / f"{image_path.stem}.jpg"
        if output_path.exists():
            continue

        input_image = Image.open(image_path).convert("RGB").resize((args.width, args.height))
        refined = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=input_image,
            height=args.height,
            width=args.width,
            strength=args.strength,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            true_cfg_scale=args.true_cfg_scale,
            generator=generator,
            max_sequence_length=FLUX_MAX_SEQUENCE_LENGTH,
        ).images[0]
        save_jpeg(refined, output_path, quality=args.jpeg_quality)


if __name__ == "__main__":
    main()
