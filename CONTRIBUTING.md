# Contributing

Thanks for your interest in improving EvalPipe.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

All of these must pass — they are enforced by CI:

```bash
ruff check . && ruff format --check .   # lint + formatting
mypy                                    # strict type checking
pytest --cov --cov-fail-under=85        # tests + coverage gate
```

## Guidelines

- Keep the core dependency-light. New runtime dependencies need a strong reason.
- Every new evaluator, statistic, or loader ships with tests covering its edge cases
  (empty inputs, missing references, malformed data) — not just the happy path.
- Statistical code must be verified against published reference values in tests.
- Error messages should say *what* went wrong and *where* (file, line, item id).
- User-facing behaviour (CLI flags, API shapes, config keys) is documented in the README.

## Adding an evaluator

1. Add a config model in `config.py` and include it in the `EvaluatorConfig` union.
2. Implement the evaluator under `evaluators/` (return scores in `[0, 1]`; never raise
   for ordinary data conditions — score 0 with a `detail` instead).
3. Register it in `evaluators/__init__.py::build_evaluators`.
4. Add tests in `tests/test_evaluators.py`.
