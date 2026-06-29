---
name: refactor-synth-pipeline
description: Deeply understand and restructure the synthetic face generation pipeline (Vec2Face+ and OSDFace) inside the existing llm_spark_exp package, with modern Python best practices
---

# Synthetic Data Pipeline Refactoring Skill

You are helping restructure the synthetic face generation pipeline that already lives
inside the `llm-spark-exp` repository. The pipeline combines:
- **Vec2Face+**: Generates synthetic face images from embedding centers
- **OSDFace (ODSFace)**: Improves/restores generated faces
- **AdaFace/AuraFace**: Embedding extraction for identity-consistency checks

The goal is 500k identities × 50 images/identity for training AuraFace (face recognition, KYC).

## Ground truth about this repo (read before doing anything)

This is **not** a greenfield project. Work *within* the existing package — do **not** run
`uv init` or create a parallel `synth-faces/` project; that would clobber the existing
`pyproject.toml` and packaging.

- Package root: `src/llm_spark_exp/`
- Pipeline package: `src/llm_spark_exp/synthetic_faces/`
- Python: **3.13** (`requires-python = ">=3.13,<3.14"`, ruff `target-version = "py313"`)
- Build backend: **hatchling** (`[tool.hatch.build.targets.wheel] packages = ["src/llm_spark_exp"]`)
- Dependency manager: **uv** with `[project.optional-dependencies]` and `[dependency-groups]`
- Pipeline deps already declared in the `synthetic-faces` optional-dependency group
- Already configured in `pyproject.toml`: ruff, pytest, coverage; `dev` group has
  pytest, pytest-cov, hypothesis, pre-commit, ruff
- CLI already exists: `src/llm_spark_exp/synthetic_faces_cli.py` (Typer), registered as
  `synthetic-faces` in `[project.scripts]`

Current files in `src/llm_spark_exp/synthetic_faces/`:
- `center_sampling.py`, `center_generation_worker.py`
- `vec2face_plus.py`
- `face_restore.py`, `face_restore_worker.py`
- `flux_refine_worker.py`
- `adaface_filter.py`
- `pose_augment.py`, `photometric_augment.py`

Related scripts in `scripts/`: `generate_pose_landmark_bank.py`,
`extract_pose_landmark_bank_from_lmdb.py`, `run_osdface_pipeline.py`.

There is a sibling pipeline `src/llm_spark_exp/synthetic_perplexity/` — **out of scope**
for this skill unless I say otherwise.

---

## Phase 1: Deep Understanding (do this FIRST)

1. Recursively explore `src/llm_spark_exp/synthetic_faces/`, `synthetic_faces_cli.py`,
   and the related files in `scripts/`.
2. Read every relevant Python file and any notebooks/configs they depend on.
3. Identify:
   - Entry points (`synthetic_faces_cli.py`, the `scripts/` entries, any notebooks)
   - Data flow: centers → vec2face generation → OSDFace restoration → (optional flux
     refine) → QA/filtering (AdaFace) → augmentation → output
   - Dependencies actually imported vs. what the `synthetic-faces` group declares
   - Hardcoded paths, magic numbers, dead code, duplicated logic across `*_worker.py`
   - What works vs. what is half-finished
4. Produce a written summary of current state:
   - Architecture overview in text
   - Module dependency list (intra-package + external)
   - Pain points and tech-debt list

**Stop and ask me to confirm your understanding before proceeding to Phase 2.**

---

## Phase 2: Propose New Structure

Reorganize *inside* the existing package. Propose subpackages under
`src/llm_spark_exp/synthetic_faces/` following this template (adapt names to what you
actually find in Phase 1):

    src/llm_spark_exp/synthetic_faces/
    ├── __init__.py
    ├── config.py                 # Pydantic settings replacing hardcoded paths/constants
    ├── generation/
    │   ├── __init__.py
    │   ├── center_sampling.py    # <- center_sampling.py
    │   ├── center_worker.py      # <- center_generation_worker.py
    │   └── generator.py          # <- vec2face_plus.py
    ├── restoration/
    │   ├── __init__.py
    │   ├── restore.py            # <- face_restore.py
    │   ├── restore_worker.py     # <- face_restore_worker.py
    │   └── flux_refine_worker.py # <- flux_refine_worker.py
    ├── quality/
    │   ├── __init__.py
    │   └── identity_consistency.py  # <- adaface_filter.py
    ├── augment/
    │   ├── __init__.py
    │   ├── pose.py               # <- pose_augment.py
    │   └── photometric.py        # <- photometric_augment.py
    └── data/
        ├── __init__.py
        └── lmdb_io.py            # extracted LMDB read/write helpers (see scripts/)

The Typer CLI stays at `src/llm_spark_exp/synthetic_faces_cli.py` (it is the registered
`synthetic-faces` entry point); update its imports to the new module paths. Keep
thin orchestration scripts in `scripts/` but have them import from the package rather
than holding logic.

Present the explicit mapping of every current file → new location. **Stop and ask for my
approval before proceeding to Phase 3.**

---

## Phase 3: Setup / Augment Tooling

**Do NOT run `uv init`.** ruff, pytest, coverage, pre-commit, and hypothesis are already
configured. Only *add* what is missing, and keep settings consistent with the existing
`pyproject.toml` (line-length 100, `target-version = "py313"`).

If we agree to add static type checking and security linting, append these sections to
`pyproject.toml` (do not duplicate or override existing `[tool.ruff]`, `[tool.pytest.ini_options]`,
or `[tool.coverage.*]` blocks):

```toml
[tool.mypy]
python_version = "3.13"
strict = true
ignore_missing_imports = true
files = ["src/llm_spark_exp/synthetic_faces"]

[tool.bandit]
exclude_dirs = ["tests"]
skips = ["B101"]
```

Optionally widen ruff lint selects for the new code (propose before applying), e.g. add
`"S"`, `"A"`, `"RET"`, `"TCH"`, `"ARG"`, `"PTH"` to the existing `select` list, with
`"S101"` ignored for tests via `per-file-ignores`.

Add the missing dev dependencies via uv (these are *not* yet in the `dev` group):

```bash
uv add --group dev mypy bandit
```

The pipeline runtime deps are already in the `synthetic-faces` optional group — only run
`uv add --optional synthetic-faces <pkg>` if Phase 1 surfaces an import that isn't
declared there. Search online for the latest compatible version before pinning.

Create `.pre-commit-config.yaml` at the repo root if it does not already exist (check
first). Pin `rev`s to current releases — verify them online rather than trusting these:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.9.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/PyCQA/bandit
    rev: 1.8.0
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]
        additional_dependencies: ["bandit[toml]"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.14.0
    hooks:
      - id: mypy
        additional_dependencies: [numpy, pydantic]
        args: [--config-file=pyproject.toml]

  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.22.0
    hooks:
      - id: gitleaks
```

Then:

```bash
uv run pre-commit install
```

---

## Phase 4: Refactor

Move code file-by-file into the new structure:
- Extract logic from scripts/notebooks into the proper package modules; leave scripts as
  thin entry points that import from `llm_spark_exp.synthetic_faces`
- Add type hints to all function signatures
- Replace hardcoded paths with a Pydantic config class (`synthetic_faces/config.py`);
  prefer reusing `src/llm_spark_exp/paths.py` for repo-relative locations
- Replace `print()` with stdlib `logging` or `rich.console`
- Add docstrings in Google style
- Update `synthetic_faces_cli.py` and `[project.scripts]` if any entry points move

After each move, update imports across the package, the CLI, the `scripts/`, and the
tests in `tests/test_synthetic_faces.py`, and run `uv run pytest tests/test_synthetic_faces.py`.

**Never delete code without showing me what is being removed. Preserve all logic —
refactoring only, no behavior changes unless I explicitly approve.**

---

## Phase 5: Tests

Tests currently live flat in `tests/` (e.g. `tests/test_synthetic_faces.py`). You may
introduce `tests/unit/`, `tests/integration/`, `tests/property/` subdirectories, but keep
`testpaths = ["tests"]` working and don't break existing tests.

### Unit tests

Core functions per module with mocked I/O (no real model weights, no GPU).

### Property-based tests (Hypothesis — already a dev dependency)

Target mathematical invariants:
- Embedding normalization: output L2 norm is always 1.0
- Center interpolation stays on the hypersphere
- Deduplication is idempotent (running twice == running once)
- Pose angles remain within valid degree ranges after any transform
- Identity-consistency scores are in [0.0, 1.0]

Example pattern:

```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=512, max_size=512))
def test_embedding_normalization(raw_vec):
    emb = normalize_embedding(raw_vec)
    assert abs(sum(x**2 for x in emb) - 1.0) < 1e-5
```

### Integration test

End-to-end on a tiny fixture: 2 centers → generate 3 images each → QA filter → assert
output structure. Mark GPU/slow paths with markers (register them in
`[tool.pytest.ini_options].markers` since `--strict-markers` is enabled).

---

## Additional Suggestions

Search online for latest versions before installing. Consider for this ML pipeline:

- **dvc**: Track large model checkpoints and generated dataset artifacts without git bloat
- **beartype**: Runtime type checking — catches tensor shape mismatches mypy cannot see statically
- **nox or just**: Task runner for common commands (generate, test, lint, package-dataset)
- **rich**: Progress bars / structured output for long-running generation jobs (already a dependency)

(`pydantic`, `pyyaml`, `typer`, `tqdm`, `lmdb`, `omegaconf` are already available — prefer
them over adding new config/serialization libraries.)

---

## Hard Rules for This Task

1. Never delete code without showing me what is being removed first
2. Preserve all logic — this is a refactoring, not a rewrite
3. Work inside `src/llm_spark_exp/`; never run `uv init` or create a parallel project
4. Do not duplicate or override existing `pyproject.toml` tool config — only add missing sections
5. If you find bugs during Phase 1, note them in a separate list; do not fix them during refactoring
6. Commit after each phase with a descriptive commit message
7. If any step is ambiguous, ask before acting
