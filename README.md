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

## Stock Price Agent

The first agentic workflow collects delayed US stock quotes and appends normalized rows to
`data/processed/stock_prices.csv`. Each new row includes the latest price, previous saved price,
absolute price change, percent change, and an optional LLM summary of the run.

I recommend Stooq for this first NYSE experiment because its CSV endpoint is simple, free to call
without an API key, and good enough for delayed quote collection. Use a paid market-data API later
if you need official real-time exchange data, service-level guarantees, or richer fundamentals.

Run the default NYSE watchlist:

```bash
uv run stock-prices
```

Run a custom set of US ticker symbols:

```bash
uv run stock-prices IBM KO JPM
```

Use the local Ollama planner to turn a natural-language request into symbols before fetching:

```bash
uv run stock-prices --request "check big US bank stocks"
```

When `--request` is used, the same local Ollama model also summarizes the fetched prices and
their changes versus the last saved row for each symbol.

The default planning model is `gemma4:e4b`. Override it when needed:

```bash
uv run stock-prices --request "check large AI infrastructure stocks" --model gemma4:e4b
```

Choose a different output table:

```bash
uv run stock-prices IBM KO JPM --table data/processed/my_prices.csv
```

## Practices

- Move reusable notebook code into `src/llm_spark_exp`.
- Keep large data, checkpoints, and generated artifacts out of git.
- Prefer scripts or documented commands for reproducible downloads and training runs.
- Keep secrets in `.env`; use `.env.example` for variable names only.
- Add tests when code graduates from exploration to shared utilities.
