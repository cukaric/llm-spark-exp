# Synthetic Perplexity Pipeline Handoff

Last updated: 2026-06-11

This document captures the current state of the separate Perplexity-style
pipeline for building a WebFace4M-class synthetic face dataset:

- Target scale: 500,000 identities x 50 final images = 25,000,000 images.
- Strategy: create a larger reserve pool of centers, qualify centers cheaply,
  oversample per identity, then hard-gate and select exactly 50 images.
- Code boundary: new orchestration lives under `llm_spark_exp.synthetic_perplexity`;
  expensive generation and AdaFace filtering reuse the existing `synthetic_faces`
  workers and CLI.

The pipeline is intentionally not a single full-scale "run everything" command.
It writes deterministic artifacts, manifests, and shell scripts for staged
execution, because 25M images needs audit points and replacement-center logic.

## Current Repo State

New files added for this pipeline:

- `src/llm_spark_exp/synthetic_perplexity/__init__.py`
- `src/llm_spark_exp/synthetic_perplexity/pipeline.py`
- `src/llm_spark_exp/synthetic_perplexity_cli.py`
- `tests/test_synthetic_perplexity.py`
- `docs/synthetic_perplexity_pipeline.md`

Project entry point added:

```toml
synthetic-perplexity = "llm_spark_exp.synthetic_perplexity_cli:app"
```

Important local assets already present:

- `models/vec2face_plus/repo/center_feature_examples.npy`
- `models/vec2face_plus/repo/vec2face_plus/main_model.pth`
- `models/vec2face_plus/repo/weights/arcface-r100-glint360k.pth`
- `models/vec2face_plus/repo/generated_images/`
- `models/vec2face_plus/repo/generated_images_ref/`
- `data/processed/sampled_centers/random_empirical_centers_8.npy`
- `data/processed/sampled_centers/random_pca_centers_8.npy`
- `data/processed/scout_smoke_4id_x8_cuda_reports/center_scout_summary.csv`
- `data/processed/scout_smoke_4id_x8_cuda_osdface_reports/center_scout_summary.csv`

Verification already run:

```bash
.venv/bin/python -m ruff check pyproject.toml src/llm_spark_exp/synthetic_perplexity src/llm_spark_exp/synthetic_perplexity_cli.py tests/test_synthetic_perplexity.py
.venv/bin/python -m pytest tests/test_synthetic_faces.py tests/test_synthetic_perplexity.py
PYTHONPATH=src .venv/bin/python -m llm_spark_exp.synthetic_perplexity_cli --help
```

Result: lint passed, CLI import passed, and `31 passed`.

## Center-Space Conclusion

The current 10k seed centers from the Vec2Face repo should be treated as
ArcFace-R100 Glint360K vectors, which is the generator space expected by
Vec2Face/Vec2Face+.

Evidence used:

- HaiyuWu confirmed in Vec2Face issue #3 that he used ArcFace-R100 trained with
  Glint360K for feature extraction:
  `https://github.com/HaiyuWu/Vec2Face/issues/3#issuecomment-2374007786`
- `BooBooWu/Vec2Face` publishes `center_feature_examples.npy` alongside
  `weights/arcface-r100-glint360k.pth`:
  `https://huggingface.co/BooBooWu/Vec2Face/tree/main`
- The Vec2Face README uses `center_feature_examples.npy` as the example center
  file for center/ID feature generation.

Caveat: this conclusion applies to the upstream `center_feature_examples.npy`.
Any custom center file still needs its extraction metadata verified before using
ArcFace-space cosine thresholds.

## Implemented Strategy Defaults

Core config: `PerplexityScaleConfig` in
`src/llm_spark_exp/synthetic_perplexity/pipeline.py`.

Current defaults:

- `target_identities = 500_000`
- `reserve_centers = 750_000`
- `gaussian_candidates = 1_500_000`
- `inter_center_max_cosine = 0.30`
- `final_images_per_identity = 50`
- `raw_candidates_per_identity = 90`
- `max_raw_attempts_per_identity = 120`
- `min_accepted_before_select = 60`
- `anchor_attempts = 3`
- `anchor_min_similarity = 0.90`
- `anchor_min_quality = 26.0`
- `variant_min_similarity = 0.70`
- `variant_min_quality = 24.0`
- `adaface_anchor_similarity = 0.95`
- `adaface_variant_similarity = 0.95`
- `osdface_cap_fraction = 0.20`
- `shard_size = 10_000`

Default final-image quota:

| Stage | Mode | Raw attempts | Final images | Status |
| --- | --- | ---: | ---: | --- |
| `bulk_mild` | center feature | 48 | 24 | runnable |
| `moderate_yaw` | pose control | 24 | 15 | manifest only |
| `large_yaw` | pose control | 12 | 7 | manifest only |
| `extreme_yaw` | pose control | 3 | 1 | manifest only |
| `other_variation` | AttrOP | 6 | 3 | manifest only |

The runnable part today is center-feature generation plus AdaFace selection.
Pose-control and AttrOP are represented in production manifests so they are not
forgotten, but they need additional wiring before full execution.

## Pipeline Phases

### 1. Analyze Seed Centers

Purpose:

- Confirm `.npy` shape and dtype.
- Inspect norm distribution.
- Estimate sampled pairwise cosine collision risk.
- Record generator-space evidence into a JSON report.

Command:

```bash
PYTHONPATH=src .venv/bin/python -m llm_spark_exp.synthetic_perplexity_cli analyze-centers \
  models/vec2face_plus/repo/center_feature_examples.npy \
  --output data/processed/synthetic_perplexity/center_feature_examples_report.json \
  --sample-size 2048
```

### 2. Expand Centers

Purpose:

- Keep the upstream seed centers first.
- Sample additional candidate centers from a diagonal empirical Gaussian fit to
  the seed features.
- Greedily accept candidates only when nearest-center cosine is <= 0.30.
- Use FAISS for production-scale nearest-neighbor search when installed.

Small smoke command:

```bash
PYTHONPATH=src .venv/bin/python -m llm_spark_exp.synthetic_perplexity_cli expand-centers \
  models/vec2face_plus/repo/center_feature_examples.npy \
  data/processed/synthetic_perplexity/smoke_centers_32.npy \
  --target-count 32 \
  --candidate-count 128 \
  --max-cosine 0.30 \
  --backend numpy
```

Production-shape command:

```bash
PYTHONPATH=src .venv/bin/python -m llm_spark_exp.synthetic_perplexity_cli expand-centers \
  models/vec2face_plus/repo/center_feature_examples.npy \
  data/processed/synthetic_perplexity/reserve_centers_750k.npy \
  --target-count 750000 \
  --candidate-count 1500000 \
  --max-cosine 0.30 \
  --backend auto \
  --batch-size 2048
```

Production note: this needs FAISS for realistic runtime. Without FAISS, the
pipeline only allows small NumPy smoke runs and raises for huge comparisons.

### 3. Write Anchor Scout Plan

Purpose:

- Split accepted reserve centers into shards.
- Generate command scripts for 3 low-variation anchor attempts per center.
- Generate command scripts for AdaFace-based center scout scoring.

Command:

```bash
PYTHONPATH=src .venv/bin/python -m llm_spark_exp.synthetic_perplexity_cli write-anchor-plan \
  data/processed/synthetic_perplexity/reserve_centers_750k.npy \
  data/processed/synthetic_perplexity/anchor_plan \
  --shard-size 10000 \
  --python .venv/bin/python \
  --batch-size 64 \
  --device cuda
```

Artifacts:

- `anchor_plan/strategy_config.json`
- `anchor_plan/center_feature_report.json`
- `anchor_plan/shards/anchor/*.npy`
- `anchor_plan/shards/anchor/*.csv`
- `anchor_plan/shards/anchor_shards.csv`
- `anchor_plan/commands/01_generate_anchor_scout.sh`
- `anchor_plan/commands/02_score_anchor_scout.sh`
- `anchor_plan/commands/03_collect_anchor_qualifications.sh`

Run order:

```bash
bash data/processed/synthetic_perplexity/anchor_plan/commands/01_generate_anchor_scout.sh
bash data/processed/synthetic_perplexity/anchor_plan/commands/02_score_anchor_scout.sh
bash data/processed/synthetic_perplexity/anchor_plan/commands/03_collect_anchor_qualifications.sh
```

Output after collection:

- `anchor_plan/qualified/qualified_centers.npy`
- `anchor_plan/qualified/qualified_centers.csv`
- `anchor_plan/qualified/qualified_centers.summary.json`

### 4. Write Production Plan

Purpose:

- Take qualified centers.
- Split into production shards.
- Write runnable center-feature generation commands for the `bulk_mild` stage.
- Write manifest placeholders for pose-control and AttrOP stages.
- Write merge and AdaFace top-50 selection scripts.

Command:

```bash
PYTHONPATH=src .venv/bin/python -m llm_spark_exp.synthetic_perplexity_cli write-production-plan \
  data/processed/synthetic_perplexity/anchor_plan/qualified/qualified_centers.npy \
  data/processed/synthetic_perplexity/production_plan \
  --shard-size 10000 \
  --python .venv/bin/python \
  --batch-size 64 \
  --device cuda
```

Artifacts:

- `production_plan/strategy_config.json`
- `production_plan/center_feature_report.json`
- `production_plan/production_stage_manifest.csv`
- `production_plan/production_runs.csv`
- `production_plan/shards/production/*.npy`
- `production_plan/shards/production/*.csv`
- `production_plan/commands/04_generate_production_center_features.sh`
- `production_plan/commands/04b_pose_attrop_stage_manifest.sh`
- `production_plan/commands/05_merge_center_feature_candidates.sh`
- `production_plan/commands/06_filter_top50_adaface.sh`

Run order for currently runnable center-feature path:

```bash
bash data/processed/synthetic_perplexity/production_plan/commands/04_generate_production_center_features.sh
bash data/processed/synthetic_perplexity/production_plan/commands/05_merge_center_feature_candidates.sh
bash data/processed/synthetic_perplexity/production_plan/commands/06_filter_top50_adaface.sh
```

## Output Layout

Recommended root:

```text
data/processed/synthetic_perplexity/
  center_feature_examples_report.json
  reserve_centers_750k.npy
  reserve_centers_750k.summary.json
  anchor_plan/
    strategy_config.json
    center_feature_report.json
    shards/
    commands/
    reports/
    qualified/
  production_plan/
    strategy_config.json
    center_feature_report.json
    production_stage_manifest.csv
    production_runs.csv
    shards/
    commands/
    candidates/
    final/
```

Vec2Face+ generation output is still written by the existing worker under:

```text
models/vec2face_plus/repo/generated_images/<run_name>/
```

The merge step symlinks those generated images into a source-indexed candidate
tree by default. Use `--copy` only if symlinks are undesirable.

## What The Code Does Today

Implemented and tested:

- Center `.npy` loading and shape validation.
- Center-feature analysis report.
- Empirical diagonal Gaussian candidate sampling.
- Greedy cosine-separated center acceptance.
- FAISS backend selection for large runs when FAISS is installed.
- NumPy backend for small smoke runs.
- Anchor shard and command generation.
- Anchor qualification collection from `center_scout_summary.csv`.
- Production shard and run manifest generation.
- Runnable production command generation for center-feature `bulk_mild`.
- Merge of generated center-feature run folders into one candidate tree.
- AdaFace top-50 filtering command generation.

Tests cover:

- 50-image quota arithmetic.
- Generator-space evidence in center reports.
- Gaussian norm-regime preservation.
- Cosine collision rejection.
- Anchor plan creation and qualification collection.
- Production plan creation and pose/AttrOP manifest placeholders.

## Known Gaps

These should be handled before trusting a full 500k x 50 production dataset.

1. Generator-space post-generation gates

   The strategy calls for Vec2Face/ArcFace-R100 Glint360K gates:
   strict anchor `Sim > 0.90, Q > 26`, variant `Sim > 0.70, Q > 24`.
   The current staged implementation uses AdaFace scout/filtering but does not
   yet compute generator-space ArcFace similarity and MagFace/quality gates for
   every generated image.

2. Face detection and landmark sanity

   The strategy calls for single-face detection and landmark sanity checks before
   identity selection. Add a detector/landmark scoring pass and store image-level
   fail reasons.

3. AuraFace validator

   AuraFace should be added as an independent deployed-matcher validation pass.
   It should not drive generation, but it should identify identities/images that
   are clean under ArcFace/AdaFace and noisy under the target matcher.

4. Pose-control wiring

   Production manifests include pose-control stages, but there is no final
   per-identity anchor image and landmark selection path yet. Next step is to
   choose accepted anchor images, map them to source center IDs, then emit
   Vec2Face+ pose commands per shard or per pose bank.

5. AttrOP implementation

   The `other_variation` quota is represented, but no runnable AttrOP command is
   emitted yet. Keep this low priority and low quota until throughput and
   identity consistency are proven.

6. OSDFace tail restoration

   OSDFace should be used only for identity-valid but visually soft images, then
   reverified after restoration. The repo has OSDFace helper code, but this
   Perplexity pipeline does not yet select the tail subset or rejoin restored
   images into final quotas.

7. Replacement-center loop

   The plan can collect qualified centers, but it does not yet automate "if an
   identity fails to reach 60 accepted after 120 attempts, drop it and replace
   with a reserve center." That needs a production audit report and a reserve
   refill command.

8. Full-scale storage and monitoring

   A 25M-image run needs shard-level progress, disk accounting, failure counts,
   and resumability. The current shell scripts are a first control plane, not a
   scheduler.

## Recommended Next Work

Best next engineering order:

1. Run a small end-to-end smoke with 32 to 128 centers:
   `expand-centers`, `write-anchor-plan`, generated anchor commands, qualification
   collection, `write-production-plan`, center-feature generation, merge, filter.
2. Add generator-space ArcFace/MagFace post-generation scoring.
3. Add face detection and landmark sanity reporting.
4. Add production audit summaries:
   per identity raw attempts, detector pass count, ArcFace pass count, AdaFace
   pass count, final selected count, and fail reason.
5. Add reserve replacement logic.
6. Add pose-control anchor mapping and command emission.
7. Add AuraFace validation.
8. Add OSDFace tail selection and reverify loop.
9. Benchmark full `expand-centers` with FAISS and tune candidate count before
   committing to 1.5M candidates.

## Useful Smoke Commands

Tiny planning-only smoke:

```bash
PYTHONPATH=src .venv/bin/python -m llm_spark_exp.synthetic_perplexity_cli expand-centers \
  models/vec2face_plus/repo/center_feature_examples.npy \
  data/processed/synthetic_perplexity/smoke_centers_32.npy \
  --target-count 32 \
  --candidate-count 128 \
  --backend numpy

PYTHONPATH=src .venv/bin/python -m llm_spark_exp.synthetic_perplexity_cli write-anchor-plan \
  data/processed/synthetic_perplexity/smoke_centers_32.npy \
  data/processed/synthetic_perplexity/smoke_anchor_plan \
  --shard-size 8 \
  --python .venv/bin/python \
  --batch-size 8 \
  --device cuda

PYTHONPATH=src .venv/bin/python -m llm_spark_exp.synthetic_perplexity_cli write-production-plan \
  data/processed/synthetic_perplexity/smoke_centers_32.npy \
  data/processed/synthetic_perplexity/smoke_production_plan \
  --shard-size 8 \
  --python .venv/bin/python \
  --batch-size 8 \
  --device cuda
```

Tests:

```bash
.venv/bin/python -m pytest tests/test_synthetic_perplexity.py
.venv/bin/python -m pytest tests/test_synthetic_faces.py tests/test_synthetic_perplexity.py
.venv/bin/python -m ruff check pyproject.toml src/llm_spark_exp/synthetic_perplexity src/llm_spark_exp/synthetic_perplexity_cli.py tests/test_synthetic_perplexity.py
```

## Quick Decision Notes

- The separate package approach is good. It keeps 500k-scale orchestration out
  of the older `synthetic_faces` helpers while still reusing generation and
  filtering code.
- The current center expansion sampler is a practical placeholder. Before a
  huge run, consider replacing or augmenting the diagonal Gaussian with PCA or a
  covariance-aware sampler calibrated against the real WebFace4M feature
  distribution.
- Do not loosen thresholds just to hit 50 images per identity. The intended
  behavior is to drop weak identities and pull from reserve centers.
- Keep generated manifests even for stages that are not runnable yet. They make
  the quota math and missing pieces visible.
