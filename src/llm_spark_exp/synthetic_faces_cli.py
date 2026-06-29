"""Command-line entry point for synthetic face generation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from llm_spark_exp.synthetic_faces import (
    ADAFACE_MODEL_ID,
    DEFAULT_CENTER_FEATURES_PATH,
    DEFAULT_VEC2FACE_PLUS_REPO_URL,
    HSFACE_DATASETS,
    VFACE_DATASETS,
    CenterScoutThresholds,
    Vec2FacePlusPaths,
    Vec2FacePlusRunner,
    augment_identity_dataset,
    augment_pose_identity_dataset,
    build_center_sampling_plan,
    default_vec2face_plus_weights,
    download_center_feature_examples,
    download_hsface_lmdb,
    download_vface_lmdb,
    export_center_feature_subsets,
    export_lmdb_images,
    filter_identity_dataset,
    load_center_features,
    load_center_scout_summary_csv,
    score_center_scout_dataset,
    write_center_sampling_plan_csv,
    write_generation_commands,
)
from llm_spark_exp.synthetic_faces.generation.vec2face_plus import (
    build_huggingface_download_commands,
)

app = typer.Typer(help="Synthetic face generation workflows.")
console = Console()
DEFAULT_PATHS = Vec2FacePlusPaths()


@app.command("vec2face-plus-plan")
def vec2face_plus_plan(
    repo_dir: Annotated[
        Path,
        typer.Option("--repo-dir", help="Local checkout of the Vec2Face+ repository."),
    ] = DEFAULT_PATHS.repo_dir,
    weights_dir: Annotated[
        Path,
        typer.Option("--weights-dir", help="Directory where Vec2Face+ weights are stored."),
    ] = DEFAULT_PATHS.weights_dir,
) -> None:
    """Print setup locations and weight download snippets for Vec2Face+."""

    console.print(f"[bold]Repository:[/bold] {DEFAULT_VEC2FACE_PLUS_REPO_URL}")
    console.print(f"[bold]Expected checkout:[/bold] {repo_dir}")
    console.print(f"[bold]Expected weights:[/bold] {weights_dir}")
    console.print("[bold]Weight downloads:[/bold]")
    for command in build_huggingface_download_commands(
        default_vec2face_plus_weights(weights_dir=weights_dir)
    ):
        console.print(f"python -c {command!r}")


@app.command("generate-with-pose")
def generate_with_pose(
    image_file: Annotated[
        Path,
        typer.Argument(help="Aligned reference image file or input file accepted by Vec2Face+."),
    ],
    pose_file: Annotated[
        Path,
        typer.Argument(help="Pose landmark file accepted by Vec2Face+."),
    ],
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Vec2Face+ output run name."),
    ] = "synthetic_faces",
    repo_dir: Annotated[
        Path,
        typer.Option("--repo-dir", help="Local checkout of the Vec2Face+ repository."),
    ] = DEFAULT_PATHS.repo_dir,
    weights_dir: Annotated[
        Path,
        typer.Option("--weights-dir", help="Directory where Vec2Face+ weights are stored."),
    ] = DEFAULT_PATHS.weights_dir,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="Vec2Face+ batch size."),
    ] = 64,
    examples: Annotated[
        int,
        typer.Option("--examples", min=1, help="Number of generated examples per input."),
    ] = 1,
    lora_r: Annotated[
        int,
        typer.Option("--lora-r", min=1, help="LoRA rank for pose control."),
    ] = 8,
    use_lora: Annotated[
        bool,
        typer.Option("--use-lora/--no-use-lora", help="Enable Vec2Face+ LoRA pose control."),
    ] = True,
    python_executable: Annotated[
        str,
        typer.Option("--python", help="Python executable inside the Vec2Face+ environment."),
    ] = "python",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the command without executing it."),
    ] = False,
) -> None:
    """Generate synthetic faces with Vec2Face+ pose control."""

    paths = Vec2FacePlusPaths(
        repo_dir=repo_dir,
        weights_dir=weights_dir,
        python_executable=python_executable,
    )
    runner = Vec2FacePlusRunner(paths)
    command = runner.build_pose_command(
        image_file=image_file,
        pose_file=pose_file,
        name=name,
        batch_size=batch_size,
        examples=examples,
        use_lora=use_lora,
        lora_r=lora_r,
    )

    if dry_run:
        console.print(" ".join(command))
        return

    try:
        run = runner.generate_with_pose(
            image_file=image_file,
            pose_file=pose_file,
            name=name,
            batch_size=batch_size,
            examples=examples,
            use_lora=use_lora,
            lora_r=lora_r,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        console.print(f"[red]Synthetic face generation failed:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(f"[green]Generated synthetic faces:[/green] {run.output_dir}")


@app.command("download-center-features")
def download_center_features(
    weights_dir: Annotated[
        Path,
        typer.Option("--weights-dir", help="Directory where center-feature vectors are stored."),
    ] = DEFAULT_PATHS.weights_dir,
) -> None:
    """Download upstream example center-feature vectors."""

    try:
        path = download_center_feature_examples(local_dir=weights_dir)
    except RuntimeError as error:
        console.print(f"[red]Center-feature download failed:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(f"[green]Downloaded center features:[/green] {path}")


@app.command("generate-from-center-features")
def generate_from_center_features(
    center_feature: Annotated[
        Path,
        typer.Option("--center-feature", help="Numpy array of identity center features."),
    ] = DEFAULT_CENTER_FEATURES_PATH,
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Vec2Face+ output run name."),
    ] = "center_feature_identities",
    repo_dir: Annotated[
        Path,
        typer.Option("--repo-dir", help="Local checkout of the Vec2Face+ repository."),
    ] = DEFAULT_PATHS.repo_dir,
    weights_dir: Annotated[
        Path,
        typer.Option("--weights-dir", help="Directory where Vec2Face+ weights are stored."),
    ] = DEFAULT_PATHS.weights_dir,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="Vec2Face+ batch size."),
    ] = 64,
    examples: Annotated[
        int,
        typer.Option("--examples", min=1, help="Number of generated examples per identity."),
    ] = 1,
    start_end: Annotated[
        str | None,
        typer.Option("--start-end", help="Identity slice, for example 0:50."),
    ] = None,
    device: Annotated[
        str,
        typer.Option("--device", help="Torch device to use, usually cuda."),
    ] = "cuda",
    python_executable: Annotated[
        str,
        typer.Option("--python", help="Python executable inside the Vec2Face+ environment."),
    ] = "python",
    skip_feature_refinement: Annotated[
        bool,
        typer.Option(
            "--skip-feature-refinement/--refine-features",
            help="Skip the slower identity-feature refinement pass.",
        ),
    ] = False,
    variation_sigmas: Annotated[
        str,
        typer.Option(
            "--variation-sigmas",
            help="Comma-separated intraclass noise scales; lower values preserve identity.",
        ),
    ] = "0.08,0.12,0.18",
    variation_weights: Annotated[
        str,
        typer.Option(
            "--variation-weights",
            help="Comma-separated proportions for each variation sigma.",
        ),
    ] = "0.5,0.35,0.15",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the command without executing it."),
    ] = False,
) -> None:
    """Generate new identities from center-feature vectors."""

    paths = Vec2FacePlusPaths(
        repo_dir=repo_dir,
        weights_dir=weights_dir,
        python_executable=python_executable,
    )
    runner = Vec2FacePlusRunner(paths)
    command = runner.build_center_feature_command(
        center_feature=center_feature,
        name=name,
        batch_size=batch_size,
        examples=examples,
        start_end=start_end,
        device=device,
        skip_feature_refinement=skip_feature_refinement,
        variation_sigmas=variation_sigmas,
        variation_weights=variation_weights,
    )

    if dry_run:
        console.print(" ".join(command))
        return

    try:
        run = runner.generate_from_center_features(
            center_feature=center_feature,
            name=name,
            batch_size=batch_size,
            examples=examples,
            start_end=start_end,
            device=device,
            skip_feature_refinement=skip_feature_refinement,
            variation_sigmas=variation_sigmas,
            variation_weights=variation_weights,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        console.print(f"[red]Center-feature generation failed:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(f"[green]Generated new identities:[/green] {run.output_dir}")


@app.command("download-vface")
def download_vface(
    dataset_size: Annotated[
        str,
        typer.Argument(help=f"VFace dataset size: {', '.join(sorted(VFACE_DATASETS))}."),
    ] = "10k",
    weights_dir: Annotated[
        Path,
        typer.Option("--weights-dir", help="Directory where released VFace LMDBs are stored."),
    ] = DEFAULT_PATHS.weights_dir,
) -> None:
    """Download a released Vec2Face+ VFace LMDB dataset."""

    try:
        path = download_vface_lmdb(dataset_size=dataset_size, local_dir=weights_dir)
    except (RuntimeError, ValueError) as error:
        console.print(f"[red]VFace download failed:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(f"[green]Downloaded VFace LMDB:[/green] {path}")


@app.command("download-hsface")
def download_hsface(
    dataset_size: Annotated[
        str,
        typer.Argument(help=f"HSFace dataset size: {', '.join(sorted(HSFACE_DATASETS))}."),
    ] = "10k",
    weights_dir: Annotated[
        Path,
        typer.Option("--weights-dir", help="Directory where released HSFace LMDBs are stored."),
    ] = DEFAULT_PATHS.weights_dir,
) -> None:
    """Download a released Vec2Face HSFace LMDB dataset."""

    try:
        path = download_hsface_lmdb(dataset_size=dataset_size, local_dir=weights_dir)
    except (RuntimeError, ValueError) as error:
        console.print(f"[red]HSFace download failed:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(f"[green]Downloaded HSFace LMDB:[/green] {path}")


@app.command("export-lmdb-images")
def export_lmdb(
    lmdb_path: Annotated[
        Path,
        typer.Argument(help="Path to a Vec2Face/Vec2Face+ LMDB dataset."),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Directory where exported images should be written."),
    ],
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Maximum number of images to export."),
    ] = None,
) -> None:
    """Export images from a released Vec2Face/Vec2Face+ LMDB dataset."""

    try:
        export = export_lmdb_images(lmdb_path=lmdb_path, output_dir=output_dir, limit=limit)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        console.print(f"[red]LMDB export failed:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(f"[green]Exported {export.images} images:[/green] {export.output_dir}")


@app.command("filter-with-adaface")
def filter_with_adaface(
    source_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing one subfolder per identity."),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Directory where filtered identity folders should be written."),
    ],
    keep_per_identity: Annotated[
        int,
        typer.Option("--keep-per-identity", min=1, help="Images to keep per identity."),
    ] = 50,
    model_id: Annotated[
        str,
        typer.Option("--model-id", help="Hugging Face AdaFace/CVLFace model id."),
    ] = ADAFACE_MODEL_ID,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="AdaFace embedding batch size."),
    ] = 64,
    device: Annotated[
        str,
        typer.Option("--device", help="Torch device for AdaFace inference."),
    ] = "cuda",
    anchor_count: Annotated[
        int,
        typer.Option("--anchor-count", min=1, help="Top samples used to build identity centroid."),
    ] = 8,
    quality_weight: Annotated[
        float,
        typer.Option("--quality-weight", help="Weight for AdaFace norm quality in ranking."),
    ] = 0.05,
    diversity_weight: Annotated[
        float,
        typer.Option(
            "--diversity-weight",
            help="Penalty for selecting images too similar to already-selected images.",
        ),
    ] = 0.0,
    min_similarity: Annotated[
        float | None,
        typer.Option("--min-similarity", help="Optional hard cutoff for identity similarity."),
    ] = None,
    report_path: Annotated[
        Path | None,
        typer.Option("--report", help="CSV path for scores and selected flags."),
    ] = None,
) -> None:
    """Filter generated identity folders with AdaFace similarity and quality scores."""

    try:
        result = filter_identity_dataset(
            source_dir=source_dir,
            output_dir=output_dir,
            keep_per_identity=keep_per_identity,
            model_id=model_id,
            batch_size=batch_size,
            device=device,
            anchor_count=anchor_count,
            quality_weight=quality_weight,
            diversity_weight=diversity_weight,
            min_similarity=min_similarity,
            report_path=report_path,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        console.print(f"[red]AdaFace filtering failed:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(
        f"[green]Filtered {result.copied_images}/{result.scored_images} images "
        f"across {result.identities} identities:[/green] {result.output_dir}"
    )
    console.print(f"[bold]Report:[/bold] {result.report_path}")


@app.command("score-center-scout")
def score_center_scout(
    source_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing generated center folders."),
    ],
    report_dir: Annotated[
        Path,
        typer.Argument(help="Directory where scout reports should be written."),
    ],
    keep_per_identity: Annotated[
        int,
        typer.Option("--keep-per-identity", min=1, help="Images to mark selected per center."),
    ] = 8,
    min_similarity: Annotated[
        float,
        typer.Option("--min-similarity", help="AdaFace hard cutoff for usable samples."),
    ] = 0.95,
    min_yield_rate: Annotated[
        float,
        typer.Option("--min-yield-rate", help="Minimum usable fraction for accepting a center."),
    ] = 0.60,
    min_selected: Annotated[
        int,
        typer.Option(
            "--min-selected", min=1, help="Minimum usable samples for accepting a center."
        ),
    ] = 4,
    min_images: Annotated[
        int,
        typer.Option("--min-images", min=1, help="Minimum scored images expected per center."),
    ] = 4,
    model_id: Annotated[
        str,
        typer.Option("--model-id", help="Hugging Face AdaFace/CVLFace model id."),
    ] = ADAFACE_MODEL_ID,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="AdaFace embedding batch size."),
    ] = 64,
    device: Annotated[
        str,
        typer.Option("--device", help="Torch device for AdaFace inference."),
    ] = "cuda",
    anchor_count: Annotated[
        int,
        typer.Option("--anchor-count", min=1, help="Top samples used to build identity centroid."),
    ] = 8,
    quality_weight: Annotated[
        float,
        typer.Option("--quality-weight", help="Weight for AdaFace norm quality in ranking."),
    ] = 0.05,
    diversity_weight: Annotated[
        float,
        typer.Option("--diversity-weight", help="Penalty for selecting near-duplicate samples."),
    ] = 0.02,
) -> None:
    """Score a generated center scout run and write center-level acceptance reports."""

    thresholds = CenterScoutThresholds(
        min_similarity=min_similarity,
        min_yield_rate=min_yield_rate,
        min_selected=min_selected,
        min_images=min_images,
    )
    try:
        result = score_center_scout_dataset(
            source_dir=source_dir,
            report_dir=report_dir,
            keep_per_identity=keep_per_identity,
            thresholds=thresholds,
            model_id=model_id,
            batch_size=batch_size,
            device=device,
            anchor_count=anchor_count,
            quality_weight=quality_weight,
            diversity_weight=diversity_weight,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        console.print(f"[red]Center scout scoring failed:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(
        f"[green]Accepted {result.accepted_centers}/{result.identities} centers "
        f"from {result.scored_images} scored images.[/green]"
    )
    console.print(f"[bold]Image report:[/bold] {result.image_report_path}")
    console.print(f"[bold]Center report:[/bold] {result.summary_report_path}")


@app.command("build-center-plan")
def build_center_plan(
    center_feature: Annotated[
        Path,
        typer.Argument(help="Numpy array of Vec2Face+ center features."),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Directory where plan artifacts should be written."),
    ],
    scout_summary: Annotated[
        Path | None,
        typer.Option("--scout-summary", help="Optional center_scout_summary.csv from scoring."),
    ] = None,
    examples: Annotated[
        int,
        typer.Option("--examples", min=1, help="Candidates to generate per accepted center."),
    ] = 24,
    retry_examples: Annotated[
        int,
        typer.Option("--retry-examples", min=1, help="Candidates for rejected/unscored retries."),
    ] = 12,
    include_rejected: Annotated[
        bool,
        typer.Option(
            "--include-rejected/--accepted-only",
            help="Include rejected centers with the conservative retry schedule.",
        ),
    ] = False,
    include_unscored: Annotated[
        bool,
        typer.Option(
            "--include-unscored/--scored-only",
            help="Include centers absent from the scout report with the retry schedule.",
        ),
    ] = False,
    low_norm_quantile: Annotated[
        float,
        typer.Option("--low-norm-quantile", help="Quantile boundary for low-norm centers."),
    ] = 0.15,
    high_norm_quantile: Annotated[
        float,
        typer.Option("--high-norm-quantile", help="Quantile boundary for high-norm centers."),
    ] = 0.85,
    prefix: Annotated[
        str,
        typer.Option("--prefix", help="Prefix for exported center-feature subset files."),
    ] = "selected_centers",
    run_name_prefix: Annotated[
        str,
        typer.Option("--run-name-prefix", help="Prefix for generated Vec2Face+ run names."),
    ] = "center_plan",
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="Vec2Face+ generation batch size."),
    ] = 64,
    device: Annotated[
        str,
        typer.Option("--device", help="Torch device for Vec2Face+ generation."),
    ] = "cuda",
    python_executable: Annotated[
        str,
        typer.Option("--python", help="Python executable for both CLI and Vec2Face+ worker."),
    ] = ".venv/bin/python",
) -> None:
    """Build adaptive center-feature subsets and command snippets for scaled generation."""

    try:
        features = load_center_features(center_feature)
        summaries = load_center_scout_summary_csv(scout_summary) if scout_summary else ()
        rows = build_center_sampling_plan(
            center_features=features,
            scout_summaries=summaries,
            examples=examples,
            retry_examples=retry_examples,
            include_rejected=include_rejected,
            include_unscored=include_unscored,
            low_norm_quantile=low_norm_quantile,
            high_norm_quantile=high_norm_quantile,
        )
        if not rows:
            raise ValueError("No centers matched the requested plan filters.")

        output_dir.mkdir(parents=True, exist_ok=True)
        plan_path = output_dir / "center_sampling_plan.csv"
        write_center_sampling_plan_csv(rows, plan_path)
        subsets = export_center_feature_subsets(
            center_features=features,
            plan_rows=rows,
            output_dir=output_dir,
            prefix=prefix,
        )
        commands_path = write_generation_commands(
            subsets=subsets,
            output_path=output_dir / "generate_commands.sh",
            run_name_prefix=run_name_prefix,
            module_python=python_executable,
            vec2face_python=python_executable,
            batch_size=batch_size,
            device=device,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        console.print(f"[red]Center plan build failed:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(f"[green]Planned {len(rows)} centers across {len(subsets)} subsets.[/green]")
    console.print(f"[bold]Plan:[/bold] {plan_path}")
    for subset in subsets:
        console.print(
            f"[bold]{subset.subset_name}:[/bold] {subset.rows} centers, "
            f"examples={subset.examples}, features={subset.feature_path}"
        )
    console.print(f"[bold]Commands:[/bold] {commands_path}")


@app.command("augment-photometric")
def augment_photometric(
    source_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing one subfolder per identity."),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Directory where augmented identity folders should be written."),
    ],
    variants_per_image: Annotated[
        int,
        typer.Option("--variants-per-image", min=1, help="Augmented copies per source image."),
    ] = 1,
    include_originals: Annotated[
        bool,
        typer.Option(
            "--include-originals/--no-include-originals",
            help="Copy originals into the augmented output too.",
        ),
    ] = True,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for deterministic augmentations."),
    ] = 18,
) -> None:
    """Create mild photometric variants for aligned face crops."""

    try:
        result = augment_identity_dataset(
            source_dir=source_dir,
            output_dir=output_dir,
            variants_per_image=variants_per_image,
            include_originals=include_originals,
            seed=seed,
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]Photometric augmentation failed:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(
        f"[green]Created {result.output_images} images from {result.source_images} "
        f"sources across {result.identities} identities:[/green] {result.output_dir}"
    )


@app.command("augment-pose")
def augment_pose(
    source_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing one subfolder per identity."),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Directory where augmented identity folders should be written."),
    ],
    variants_per_image: Annotated[
        int,
        typer.Option("--variants-per-image", min=1, help="Augmented copies per source image."),
    ] = 1,
    include_originals: Annotated[
        bool,
        typer.Option(
            "--include-originals/--no-include-originals",
            help="Copy originals into the augmented output too.",
        ),
    ] = True,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for deterministic augmentations."),
    ] = 23,
) -> None:
    """Create mild pose-like geometric variants for aligned face crops."""

    try:
        result = augment_pose_identity_dataset(
            source_dir=source_dir,
            output_dir=output_dir,
            variants_per_image=variants_per_image,
            include_originals=include_originals,
            seed=seed,
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]Pose augmentation failed:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(
        f"[green]Created {result.output_images} images from {result.source_images} "
        f"sources across {result.identities} identities:[/green] {result.output_dir}"
    )


@app.command("refine-with-flux")
def refine_with_flux(
    source_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing one subfolder per identity."),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Directory where FLUX-refined identity folders should be written."),
    ],
    model_id: Annotated[
        str,
        typer.Option("--model-id", help="Hugging Face FLUX model id."),
    ] = "black-forest-labs/FLUX.1-schnell",
    prompt: Annotated[
        str,
        typer.Option("--prompt", help="Image-to-image refinement prompt."),
    ] = (
        "high quality realistic passport-style face photo, natural skin texture, "
        "sharp eyes, clean lighting, preserve the same person, preserve head pose, "
        "no beauty retouching"
    ),
    negative_prompt: Annotated[
        str | None,
        typer.Option("--negative-prompt", help="Optional negative prompt."),
    ] = None,
    strength: Annotated[
        float,
        typer.Option("--strength", min=0.0, max=1.0, help="Image-to-image denoise strength."),
    ] = 0.18,
    num_inference_steps: Annotated[
        int,
        typer.Option("--steps", min=1, help="FLUX inference steps."),
    ] = 4,
    guidance_scale: Annotated[
        float,
        typer.Option("--guidance-scale", help="FLUX guidance scale."),
    ] = 0.0,
    true_cfg_scale: Annotated[
        float,
        typer.Option("--true-cfg-scale", help="True CFG scale when supported by pipeline."),
    ] = 1.0,
    cache_dir: Annotated[
        Path,
        typer.Option("--cache-dir", help="Local Hugging Face cache for FLUX weights."),
    ] = Path("models/flux"),
    height: Annotated[
        int,
        typer.Option("--height", help="Output height."),
    ] = 512,
    width: Annotated[
        int,
        typer.Option("--width", help="Output width."),
    ] = 512,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic seed."),
    ] = 20260611,
    batch_device: Annotated[
        str,
        typer.Option("--device", help="Torch device for FLUX inference."),
    ] = "cuda",
    dtype: Annotated[
        str,
        typer.Option("--dtype", help="Torch dtype: bfloat16, float16, or float32."),
    ] = "bfloat16",
    cpu_offload: Annotated[
        bool,
        typer.Option("--cpu-offload/--no-cpu-offload", help="Enable diffusers CPU offload."),
    ] = False,
    max_images: Annotated[
        int | None,
        typer.Option("--max-images", min=1, help="Maximum images to refine."),
    ] = None,
    identity_limit: Annotated[
        int | None,
        typer.Option("--identity-limit", min=1, help="Maximum identity folders to process."),
    ] = None,
    jpeg_quality: Annotated[
        int,
        typer.Option("--jpeg-quality", min=1, max=100, help="JPEG quality."),
    ] = 95,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the worker command without executing it."),
    ] = False,
) -> None:
    """Refine identity-organized face crops using FLUX image-to-image."""

    import os
    import subprocess
    import sys

    command = [
        sys.executable,
        "-m",
        "llm_spark_exp.synthetic_faces.restoration.flux_refine_worker",
        "--source-dir",
        str(source_dir),
        "--output-dir",
        str(output_dir),
        "--model-id",
        model_id,
        "--cache-dir",
        str(cache_dir),
        "--prompt",
        prompt,
        "--strength",
        str(strength),
        "--num-inference-steps",
        str(num_inference_steps),
        "--guidance-scale",
        str(guidance_scale),
        "--true-cfg-scale",
        str(true_cfg_scale),
        "--height",
        str(height),
        "--width",
        str(width),
        "--seed",
        str(seed),
        "--device",
        batch_device,
        "--dtype",
        dtype,
        "--jpeg-quality",
        str(jpeg_quality),
    ]
    if negative_prompt is not None:
        command.extend(["--negative-prompt", negative_prompt])
    if cpu_offload:
        command.append("--cpu-offload")
    if max_images is not None:
        command.extend(["--max-images", str(max_images)])
    if identity_limit is not None:
        command.extend(["--identity-limit", str(identity_limit)])

    if dry_run:
        console.print(" ".join(command))
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as error:
        console.print(f"[red]FLUX refinement failed:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(f"[green]FLUX-refined images:[/green] {output_dir}")


if __name__ == "__main__":
    app()
