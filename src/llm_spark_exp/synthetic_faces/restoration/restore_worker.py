"""Subprocess worker for OSDFace face restoration on identity-organized folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_spark_exp.synthetic_faces._common import save_jpeg
from llm_spark_exp.synthetic_faces.constants import IMAGE_SUFFIXES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("OSDFace face restoration")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--weights-dir", required=True, type=Path)
    parser.add_argument("--sd-model", default="stabilityai/stable-diffusion-2-1-base", type=str)
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--mixed-precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--jpeg-quality", default=95, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--upscale", action="store_true")
    parser.add_argument("--upscale-factor", default=2, type=int)
    return parser.parse_args()


def load_osdface_model(
    *,
    repo_dir: Path,
    weights_dir: Path,
    sd_model: str,
    merge_lora: bool,
    mixed_precision: str,
    device: str,
):
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

    repo_dir = repo_dir.resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from lq_embed import TwoLayerConv1x1, vqvae_encoder
    from utils.others import get_x0_from_noise

    weight_dtype = torch.float16 if mixed_precision == "fp16" else torch.float32
    torch_device = torch.device(device)

    noise_scheduler = DDIMScheduler.from_pretrained(sd_model, subfolder="scheduler")
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(torch_device)
    vae = AutoencoderKL.from_pretrained(sd_model, subfolder="vae")
    vae.to(torch_device, dtype=weight_dtype)

    if merge_lora:
        unet = _merge_lora_unet(sd_model, weights_dir, repo_dir)
    else:
        from diffusers import StableDiffusionPipeline

        unet = UNet2DConditionModel.from_pretrained(sd_model, subfolder="unet")
        pipe = StableDiffusionPipeline(vae, None, None, unet, noise_scheduler, None, None)
        pipe.load_lora_weights(str(weights_dir))

    unet.to(torch_device, dtype=weight_dtype)

    args = _make_vqvae_args(str(weights_dir / "associate_2.ckpt"))
    img_encoder = vqvae_encoder(args).to(torch_device, dtype=weight_dtype)

    embedding_change = TwoLayerConv1x1(512, 1024)
    embedding_change.load_state_dict(
        torch.load(str(weights_dir / "embedding_change_weights.pth"), weights_only=False)
    )
    embedding_change.to(torch_device, dtype=weight_dtype)

    return {
        "vae": vae,
        "unet": unet,
        "img_encoder": img_encoder,
        "embedding_change": embedding_change,
        "alphas_cumprod": alphas_cumprod,
        "weight_dtype": weight_dtype,
        "device": torch_device,
        "timestep": 399,
        "get_x0_from_noise": get_x0_from_noise,
    }


def _merge_lora_unet(sd_model: str, weights_dir: Path, repo_dir: Path):
    import torch
    from diffusers import UNet2DConditionModel
    from safetensors import safe_open

    unet = UNet2DConditionModel.from_pretrained(sd_model, subfolder="unet")
    lora_rank = 16
    lora_alpha = 16
    alpha = float(lora_alpha / lora_rank)

    lora_path = str(weights_dir / "pytorch_lora_weights.safetensors")
    state_dict_unet = unet.state_dict()
    processed_keys = set()

    with safe_open(lora_path, framework="pt") as f:
        lora_state = {key: f.get_tensor(key) for key in f.keys()}

    for key in lora_state:
        if "lora_A" in key:
            lora_a_key = key
            lora_b_key = key.replace("lora_A", "lora_B")
            unet_key = key.replace(".lora_A.weight", ".weight").replace("unet.", "")
            if lora_b_key not in lora_state or unet_key not in state_dict_unet:
                continue
            W_A = lora_state[lora_a_key]
            W_B = lora_state[lora_b_key]
            original_weight = state_dict_unet[unet_key]
            processed_keys.update([lora_a_key, lora_b_key])
            if len(original_weight.shape) == 4 and len(W_A.shape) == 4 and len(W_B.shape) == 4:
                out_channels, in_channels, kH, kW = original_weight.shape
                rank = W_A.shape[0]
                W_A_flat = W_A.view(rank, -1)
                W_B_flat = W_B.view(out_channels, rank)
                delta_W_flat = torch.matmul(W_B_flat, W_A_flat)
                delta_W = delta_W_flat.view(out_channels, in_channels, kH, kW)
                merged_weight = original_weight + alpha * delta_W
            else:
                merged_weight = original_weight + alpha * torch.mm(W_B, W_A)
            state_dict_unet[unet_key] = merged_weight
        elif "lora.up.weight" in key:
            lora_up_key = key
            lora_down_key = key.replace("lora.up.weight", "lora.down.weight")
            original_weight_key = key.replace(".lora.up.weight", ".weight").replace("unet.", "")
            if lora_down_key not in lora_state or original_weight_key not in state_dict_unet:
                continue
            W_up = lora_state[lora_up_key]
            W_down = lora_state[lora_down_key]
            W_orig = state_dict_unet[original_weight_key]
            processed_keys.update([lora_up_key, lora_down_key])
            if W_orig.ndim == 2:
                delta_W = torch.matmul(W_up, W_down)
                state_dict_unet[original_weight_key] = W_orig + alpha * delta_W

    unet.load_state_dict(state_dict_unet)
    return unet


def _make_vqvae_args(img_encoder_weight: str):
    import argparse

    args = argparse.Namespace()
    args.img_encoder_weight = img_encoder_weight
    args.cat_prompt_embedding = False
    args.use_pos_embedding = False
    args.use_att_pool = False
    args.learnable_pos_emb = False
    return args


def restore_image(model: dict, image_tensor):
    import torch
    import torch.nn.functional as Fun

    with torch.no_grad():
        device = model["device"]
        dtype = model["weight_dtype"]
        lq = image_tensor.to(device, dtype=dtype)

        if lq.shape[2] != 512 or lq.shape[3] != 512:
            lq = Fun.interpolate(lq, (512, 512), mode="bilinear", align_corners=True)

        prompt_embeds = model["img_encoder"](lq).reshape(lq.shape[0], 77, -1)
        prompt_embeds = model["embedding_change"](prompt_embeds)

        lq_latent = (
            model["vae"].encode(lq).latent_dist.sample() * model["vae"].config.scaling_factor
        )

        model_pred = model["unet"](
            lq_latent, model["timestep"], encoder_hidden_states=prompt_embeds
        ).sample

        x_0 = model["get_x0_from_noise"](
            lq_latent.double(),
            model_pred.double(),
            model["alphas_cumprod"].double(),
            model["timestep"],
        ).float()

        output_image = (
            model["vae"].decode(x_0.to(dtype) / model["vae"].config.scaling_factor).sample
        ).clamp(-1, 1)
        output_image = output_image * 0.5 + 0.5

        return output_image.clamp(0.0, 1.0)


def load_swinir_model(*, device: str, upscale_factor: int = 2):
    try:
        from basicsr.archs.swinir_arch import SwinIR
    except ImportError as error:
        raise RuntimeError("basicsr is required for SwinIR upscaling.") from error

    model = SwinIR(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2,
        sf=upscale_factor,
        img_range=1.0,
        upsampler="nearest+conv",
        resi_connection="1conv",
    )
    model.eval()
    model.to(device)
    return model


def upscale_image(model, image_tensor, *, upscale_factor: int = 2):
    import torch
    import torch.nn.functional as Fun

    with torch.no_grad():
        device = next(model.parameters()).device
        inp = image_tensor.to(device)

        _, _, h, w = inp.shape
        pad_h = (upscale_factor - h % upscale_factor) % upscale_factor
        pad_w = (upscale_factor - w % upscale_factor) % upscale_factor
        if pad_h > 0 or pad_w > 0:
            inp = Fun.pad(inp, (0, pad_w, 0, pad_h), mode="reflect")

        output = model(inp)
        return output[:, :, : h * upscale_factor, : w * upscale_factor]


def main() -> None:
    args = parse_args()

    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from tqdm import tqdm

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    model = load_osdface_model(
        repo_dir=args.repo_dir,
        weights_dir=args.weights_dir,
        sd_model=args.sd_model,
        merge_lora=args.merge_lora,
        mixed_precision=args.mixed_precision,
        device=args.device,
    )

    swinir_model = None
    if args.upscale:
        swinir_model = load_swinir_model(device=args.device, upscale_factor=args.upscale_factor)

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for identity_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        images = sorted(
            path
            for path in identity_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            continue

        dest_identity_dir = output_dir / identity_dir.name
        dest_identity_dir.mkdir(parents=True, exist_ok=True)

        for image_path in tqdm(images, desc=identity_dir.name):
            output_path = dest_identity_dir / f"{image_path.stem}.jpg"
            if output_path.exists():
                continue

            input_image = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                image_tensor = TF.to_tensor(input_image).unsqueeze(0) * 2 - 1
                restored = restore_image(model, image_tensor)

                if swinir_model is not None:
                    restored = upscale_image(
                        swinir_model, restored, upscale_factor=args.upscale_factor
                    )

                restored_pil = TF.to_pil_image(restored[0].cpu())

            save_jpeg(restored_pil, output_path, quality=args.jpeg_quality)


if __name__ == "__main__":
    main()
