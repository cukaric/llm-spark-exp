# Notebooks

Use notebooks for exploration and short-lived analysis. Move reusable code into `src/llm_spark_exp`
once it is shared by more than one notebook or script.

Suggested first setup:

```bash
uv sync --group dev
uv run python -m ipykernel install --user --name llm-spark-exp --display-name "Python (llm-spark-exp)"
uv run jupyter lab
```
