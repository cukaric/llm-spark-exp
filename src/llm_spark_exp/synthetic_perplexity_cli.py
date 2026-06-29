"""CLI for the separate Perplexity-style synthetic face pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console

from llm_spark_exp.synthetic_perplexity import (
    PerplexityScaleConfig,
    analyze_center_features,
    collect_anchor_qualifications,
    expand_center_pool,
    load_center_features,
    merge_center_feature_runs,
    write_anchor_plan,
    write_production_plan,
)
from llm_spark_exp.synthetic_perplexity.pipeline import write_json

app = typer.Typer(help="Separate 500k-scale synthetic face planning pipeline.")
console = Console()


@app.command("analyze-centers")
def analyze_centers(
    center_feature: Annotated[
        Path,
        typer.Argument(help="Numpy array of Vec2Face/Vec2Face+ center features."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional JSON report path."),
    ] = None,
    sample_size: Annotated[
        int,
        typer.Option("--sample-size", min=2, help="Pairwise cosine sample size."),
    ] = 2048,
) -> None:
    """Analyze center feature shape, norms, and sampled cosine collisions."""

    try:
        features = load_center_features(center_feature)
        report = analyze_center_features(features, sample_size=sample_size)
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]Center analysis failed:[/red] {error}")
        raise typer.Exit(1) from error

    if output is not None:
        write_json(output, report)
        console.print(f"[green]Wrote center analysis:[/green] {output}")
    console.print(
        f"[green]Centers:[/green] {report['rows']} x {report['dimensions']} "
        f"({report['generator_space_evidence']['assumed_space']})"
    )


@app.command("expand-centers")
def expand_centers(
    seed_center_feature: Annotated[
        Path,
        typer.Argument(help="Seed center-feature .npy, usually center_feature_examples.npy."),
    ],
    output_feature: Annotated[
        Path,
        typer.Argument(help="Output .npy for accepted seed + sampled centers."),
    ],
    target_count: Annotated[
        int,
        typer.Option("--target-count", min=1, help="Accepted centers to keep."),
    ] = 750_000,
    candidate_count: Annotated[
        int,
        typer.Option("--candidate-count", min=1, help="Gaussian candidates to sample."),
    ] = 1_500_000,
    max_cosine: Annotated[
        float,
        typer.Option("--max-cosine", min=0.0, max=0.999, help="Max nearest-neighbor cosine."),
    ] = 0.30,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for Gaussian sampling."),
    ] = 20260611,
    backend: Annotated[
        str,
        typer.Option("--backend", help="auto, faiss, or numpy."),
    ] = "auto",
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="FAISS candidate batch size."),
    ] = 2048,
) -> None:
    """Expand seed centers using empirical Gaussian sampling and greedy cosine filtering."""

    if backend not in {"auto", "faiss", "numpy"}:
        console.print("[red]--backend must be one of: auto, faiss, numpy[/red]")
        raise typer.Exit(1)
    try:
        seed_features = load_center_features(seed_center_feature)
        result = expand_center_pool(
            seed_features,
            target_count=target_count,
            candidate_count=candidate_count,
            max_cosine=max_cosine,
            seed=seed,
            backend=backend,  # type: ignore[arg-type]
            batch_size=batch_size,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        console.print(f"[red]Center expansion failed:[/red] {error}")
        raise typer.Exit(1) from error

    output_feature.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_feature, result.features)
    write_json(output_feature.with_suffix(".summary.json"), result.summary())
    console.print(
        f"[green]Accepted {result.accepted_count} centers "
        f"({result.accepted_candidates} sampled) with {result.backend}:[/green] "
        f"{output_feature}"
    )
    if result.exhausted:
        console.print("[yellow]Candidate pool exhausted before target_count was reached.[/yellow]")


@app.command("write-anchor-plan")
def write_anchor_plan_command(
    center_feature: Annotated[
        Path,
        typer.Argument(help="Accepted center-feature pool to scout."),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Plan directory for shards, reports, and commands."),
    ],
    shard_size: Annotated[
        int,
        typer.Option("--shard-size", min=1, help="Centers per anchor shard."),
    ] = 10_000,
    python_executable: Annotated[
        str,
        typer.Option("--python", help="Python executable used in generated commands."),
    ] = ".venv/bin/python",
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="Generation and scoring batch size."),
    ] = 64,
    device: Annotated[
        str,
        typer.Option("--device", help="Torch device for generated commands."),
    ] = "cuda",
) -> None:
    """Write anchor-scout shards and command scripts."""

    config = PerplexityScaleConfig(shard_size=shard_size)
    try:
        summary = write_anchor_plan(
            center_features_path=center_feature,
            output_dir=output_dir,
            config=config,
            module_python=python_executable,
            batch_size=batch_size,
            device=device,
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]Anchor plan failed:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(
        f"[green]Anchor plan:[/green] {summary['centers']} centers, "
        f"{summary['shards']} shards in {output_dir}"
    )


@app.command("collect-anchor-qualifications")
def collect_anchor_qualifications_command(
    center_feature: Annotated[
        Path,
        typer.Argument(help="Original center-feature pool used by write-anchor-plan."),
    ],
    plan_dir: Annotated[
        Path,
        typer.Argument(help="Anchor plan directory containing reports/anchors."),
    ],
    output_feature: Annotated[
        Path,
        typer.Option("--output-feature", help="Output .npy for accepted centers."),
    ],
) -> None:
    """Collect accepted anchor centers after scout scoring."""

    try:
        summary = collect_anchor_qualifications(
            center_features_path=center_feature,
            plan_dir=plan_dir,
            output_feature_path=output_feature,
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]Qualification collection failed:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(
        f"[green]Accepted {summary['accepted_centers']}/{summary['source_centers']} centers:[/green] "
        f"{output_feature}"
    )


@app.command("write-production-plan")
def write_production_plan_command(
    center_feature: Annotated[
        Path,
        typer.Argument(help="Qualified center-feature pool for production generation."),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Plan directory for production shards and commands."),
    ],
    shard_size: Annotated[
        int,
        typer.Option("--shard-size", min=1, help="Centers per production shard."),
    ] = 10_000,
    python_executable: Annotated[
        str,
        typer.Option("--python", help="Python executable used in generated commands."),
    ] = ".venv/bin/python",
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="Generation and scoring batch size."),
    ] = 64,
    device: Annotated[
        str,
        typer.Option("--device", help="Torch device for generated commands."),
    ] = "cuda",
) -> None:
    """Write production generation, merge, and filtering command scripts."""

    config = PerplexityScaleConfig(shard_size=shard_size)
    try:
        summary = write_production_plan(
            center_features_path=center_feature,
            output_dir=output_dir,
            config=config,
            module_python=python_executable,
            batch_size=batch_size,
            device=device,
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]Production plan failed:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(
        f"[green]Production plan:[/green] {summary['centers']} centers, "
        f"{summary['center_feature_runs']} runnable center-feature runs"
    )


@app.command("merge-center-feature-runs")
def merge_center_feature_runs_command(
    plan_dir: Annotated[
        Path,
        typer.Argument(help="Production plan directory containing production_runs.csv."),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Merged source-indexed candidate output directory."),
    ],
    generated_root: Annotated[
        Path,
        typer.Option("--generated-root", help="Vec2Face+ generated_images root."),
    ] = Path("models/vec2face_plus/repo/generated_images"),
    copy_files: Annotated[
        bool,
        typer.Option("--copy/--symlink", help="Copy images instead of symlinking them."),
    ] = False,
) -> None:
    """Merge generated center-feature stage outputs into one candidate tree."""

    try:
        summary = merge_center_feature_runs(
            plan_dir=plan_dir,
            output_dir=output_dir,
            generated_root=generated_root,
            copy_files=copy_files,
        )
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]Merge failed:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(
        f"[green]Merged {summary['linked_or_copied_images']} images "
        f"for {summary['identities']} identities:[/green] {output_dir}"
    )


if __name__ == "__main__":
    app()
