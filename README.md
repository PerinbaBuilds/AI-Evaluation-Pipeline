# EvalPipe — AI Evaluation Pipeline

[![CI](https://github.com/PerinbaBuilds/AI-Evaluation-Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/PerinbaBuilds/AI-Evaluation-Pipeline/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)

**Benchmark and A/B‑test LLM outputs with real statistics — self‑hosted, dependency‑light, and runnable fully offline.**

Before you ship a new model or prompt, you need evidence it's actually better. EvalPipe runs a dataset through any model, scores every output, and answers the one question that matters: *is the candidate significantly better than the baseline, or is it just noise?*

| Dashboard | A/B comparison |
|---|---|
| ![Dashboard](docs/images/dashboard.png) | ![A/B comparison](docs/images/compare.png) |
| **Run detail** | **Real‑time playground** |
| ![Run detail](docs/images/run-detail.png) | ![Playground](docs/images/playground.png) |

## Why it's different

- **Real statistics, from first principles** — two‑proportion z‑test, Welch's t‑test, Wilson intervals, bootstrap CIs, Cohen's *h/d*, and a sample‑size calculator. No NumPy/SciPy; every function is cross‑validated against SciPy to 1e‑8.
- **Honest verdicts** — runs are paired by item id and tested at a chosen α; small samples produce an explicit *underpowered* warning, not false confidence.
- **Item‑level regression diff** — names exactly which items flipped pass→fail vs fail→pass, the failures an average hides.
- **Runs fully offline** — a deterministic simulator makes every feature (including the LLM judge) work with no API keys and no network.

## Quickstart

```bash
git clone https://github.com/PerinbaBuilds/AI-Evaluation-Pipeline.git
cd AI-Evaluation-Pipeline
python -m venv .venv && source .venv/bin/activate
pip install -e .

evalpipe demo      # seed a deterministic offline history (no keys, no network)
evalpipe serve     # dashboard → http://127.0.0.1:8000
```

Open the dashboard, then try **A/B Compare** with the two run ids `evalpipe demo` prints, or the **Playground** to compare live models on one prompt. A light/dark toggle is in the sidebar.

## Core capabilities

| Area | What you get |
|---|---|
| **Evaluators (7)** | exact match · token‑F1 · contains · regex · semantic similarity · safety blocklist · LLM‑as‑judge (correctness / faithfulness / relevance / custom) |
| **A/B testing** | paired by item id · z‑test (pass rate) + Welch's t‑test (mean score) · effect sizes · 95% CIs · bootstrap · regression diff |
| **Slice analysis** | break a run's pass rate down by any metadata field (topic, difficulty, …) with per‑group Wilson intervals |
| **Execution** | async runner, bounded concurrency, backoff retries, per‑item failure isolation, opt‑in response cache |
| **Reporting** | dashboard, per‑run charts, CSV/JSON export, Prometheus `/metrics`, health probe |
| **Ops** | latency p50/p95 + token‑cost per run, versioned prompt registry, CI quality gate (`--min-pass-rate`) |

## Define a run

```yaml
# eval.yaml
name: qa-baseline
dataset: examples/qa_dataset.jsonl
provider:
  type: openai_compatible          # works with vLLM, Ollama, llama.cpp, hosted APIs
  model: my-model
  base_url: http://localhost:11434/v1
  api_key_env: MY_PROVIDER_API_KEY # env-var NAME — secrets never live in configs
evaluators:
  - type: token_f1
    threshold: 0.6
  - type: llm_judge
    rubric: correctness
    threshold: 0.7
concurrency: 8
```

```bash
evalpipe run eval.yaml                          # run + persist
evalpipe compare <baseline_id> <candidate_id>   # statistical A/B report
evalpipe run eval.yaml --min-pass-rate 0.85     # non-zero exit if quality regresses (CI gate)
```

## Model providers

| `type` | Talks to | Cost | Key |
|---|---|---|---|
| `mock` | Deterministic offline simulator | free | — |
| `ollama` | Local models via Ollama | free (local) | — |
| `gemini` | Google Gemini `generateContent` | free tier | `GEMINI_API_KEY` |
| `groq` | Groq (fast hosted open models) | free tier | `GROQ_API_KEY` |
| `openrouter` | OpenRouter (incl. `:free` models) | free models | `OPENROUTER_API_KEY` |
| `openai` | OpenAI Chat Completions | paid | `OPENAI_API_KEY` |
| `anthropic` | Anthropic Messages API | paid | `ANTHROPIC_API_KEY` |
| `openai_compatible` | Any OpenAI-style endpoint | varies | configurable |

Keys resolve **inline‑first** (a per‑request field in the Playground) **then from the named environment variable** — and are never written to disk, logged, or returned. Free keys: [Ollama](https://ollama.com) (local), [Google AI Studio](https://aistudio.google.com/apikey), [Groq](https://console.groq.com), [OpenRouter](https://openrouter.ai).

## Deploy

Runs as a single Docker image that binds `$PORT`, so any container host works.

```bash
docker build -t evalpipe .
docker run -p 8000:8000 -e EVALPIPE_SEED_DEMO=1 evalpipe
```

**Free hosting (Koyeb):** create a service → *Deploy from GitHub* → pick this repo → Koyeb builds the `Dockerfile` → set `EVALPIPE_SEED_DEMO=1`. It boots with demo data already seeded and gives a public HTTPS URL. The health check is `/api/health`.

## Documentation

| Doc | Contents |
|---|---|
| [`docs/SRS.md`](docs/SRS.md) | Software Requirements Specification — functional & non‑functional requirements, traceability. |
| [`docs/SDS.md`](docs/SDS.md) | Software Design Specification — architecture, modules, data model, algorithms, decisions. |

## Development

```bash
pip install -e ".[dev]"
pytest --cov          # 265 tests, 93% coverage
ruff check . && ruff format --check .
mypy src               # strict
```

CI runs lint + strict typing, the test matrix on Python 3.11/3.12 with a coverage gate, and a Docker build with a live smoke test against `/api/health`.

```
src/evalpipe/
├── config.py     datasets.py     # validated configs + strict dataset loading
├── providers/    evaluators/     # model access + the metric suite
├── stats.py      ab.py           # from-scratch statistics + paired A/B
├── runner.py     pipeline.py     # async orchestration
├── storage.py    cache.py        # SQLite persistence + response cache
└── server/                        # FastAPI app, templates, SVG charts
```

## License

[MIT](LICENSE) © 2026 Perinba Athiban
