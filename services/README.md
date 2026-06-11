# Services

Use this folder when an experiment becomes something deployable, such as an API, worker, inference
server, or Dockerized tool.

For now, shared Python code should live in `src/llm_spark_exp`. Split into `packages/` or individual
service folders only when there are clear ownership, dependency, or deployment boundaries.
