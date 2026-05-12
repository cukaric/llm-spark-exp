# llm-spark-exp

Experiments with agents, local LLMs, fine-tuning, classic ML, and VLMs.

## Structure

```text
.
├── data/              # Local datasets; ignored by git except README/.gitkeep files
├── models/            # Local model artifacts/checkpoints; ignored by git by default
├── notebooks/         # Exploration and analysis notebooks
├── packages/          # Reserved for future first-party packages
├── services/          # Reserved for deployable APIs, workers, inference servers, etc.
├── src/llm_spark_exp  # Shared Python package
└── tests/             # Unit tests
```

The repo starts as one Python package under `src/llm_spark_exp`. That is usually the best shape for
early AI/ML exploration because notebooks, scripts, and tests can share code without creating a
workspace too early.

Use `services/` when something becomes deployable or Dockerized. Use `packages/` only when there are
separate dependency, release, or ownership boundaries. Until then, a single package is simpler.

## Setup

Prerequisites:

- Git
- uv
- A configured Git identity, preferably using a GitHub noreply email if GitHub email privacy is on

Create the Python 3.13 environment:

```bash
uv python install 3.13
uv sync --group dev
```

Install the notebook kernel:

```bash
uv run python -m ipykernel install --user --name llm-spark-exp --display-name "Python (llm-spark-exp)"
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

Run checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The dev environment includes Ruff, pre-commit, pytest, pytest-cov, Hypothesis, JupyterLab,
ipykernel, ipywidgets, nbclient, and nbconvert.

Start notebooks:

```bash
uv run jupyter lab
```

## Optional Experiment Stacks

Install only the extras you need for a given experiment:

```bash
uv sync --group dev --extra agents
uv sync --group dev --extra local-llm
uv sync --group dev --extra vlm
uv sync --group dev --all-extras
```

For GPU work, install PyTorch separately using the command that matches the local CUDA/ROCm/CPU
target from the official PyTorch selector. Keeping PyTorch out of the default dependencies avoids
pulling the wrong binary for a machine.

## Practices

- Move reusable notebook code into `src/llm_spark_exp`.
- Keep large data, checkpoints, and generated artifacts out of git.
- Prefer scripts or documented commands for reproducible downloads and training runs.
- Keep secrets in `.env`; use `.env.example` for variable names only.
- Add tests when code graduates from exploration to shared utilities.
