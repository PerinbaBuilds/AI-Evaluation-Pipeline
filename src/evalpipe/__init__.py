"""EvalPipe: provider-agnostic evaluation pipeline for LLM applications.

The package is organised around a small set of composable pieces:

- :mod:`evalpipe.datasets`   — dataset loading and validation (JSONL / CSV)
- :mod:`evalpipe.providers`  — model providers (mock simulation, OpenAI-compatible HTTP)
- :mod:`evalpipe.evaluators` — pluggable scoring metrics
- :mod:`evalpipe.runner`     — async orchestration with concurrency and retries
- :mod:`evalpipe.stats`      — statistical primitives implemented from first principles
- :mod:`evalpipe.ab`         — A/B comparison of two evaluation runs
- :mod:`evalpipe.storage`    — SQLite persistence
- :mod:`evalpipe.server`     — FastAPI service and reporting dashboard
"""

__version__ = "1.0.0"
