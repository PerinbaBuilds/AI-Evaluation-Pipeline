"""Offline demo seeding.

``evalpipe demo`` populates a database with a realistic evaluation history —
several nightly-style runs of a "baseline" simulated model at gradually
improving quality, plus a baseline/candidate pair suitable for the A/B
comparison view. Everything is deterministic and needs no network access or
API keys.
"""

from __future__ import annotations

from evalpipe.config import (
    EvalConfig,
    EvaluatorConfig,
    ExactMatchConfig,
    LLMJudgeConfig,
    MockProviderConfig,
    SafetyConfig,
    SemanticSimilarityConfig,
    TokenF1Config,
)
from evalpipe.pipeline import execute_run
from evalpipe.storage import Storage

DEMO_ITEMS: list[dict[str, object]] = [
    {
        "id": "qa-001",
        "prompt": "What is the capital of France?",
        "expected": "The capital of France is Paris.",
        "metadata": {
            "context": "France is a country in Western Europe. Its capital and largest city is Paris."
        },
    },
    {
        "id": "qa-002",
        "prompt": "What is the boiling point of water at sea level in Celsius?",
        "expected": "Water boils at 100 degrees Celsius at sea level.",
        "metadata": {
            "context": "At standard atmospheric pressure, water boils at 100 degrees Celsius."
        },
    },
    {
        "id": "qa-003",
        "prompt": "Who wrote the play Romeo and Juliet?",
        "expected": "Romeo and Juliet was written by William Shakespeare.",
        "metadata": {
            "context": "William Shakespeare wrote Romeo and Juliet in the early stage of his career."
        },
    },
    {
        "id": "qa-004",
        "prompt": "What planet is known as the Red Planet?",
        "expected": "Mars is known as the Red Planet.",
        "metadata": {
            "context": "Mars appears reddish because of iron oxide on its surface, earning it the name Red Planet."
        },
    },
    {
        "id": "qa-005",
        "prompt": "What is the largest ocean on Earth?",
        "expected": "The Pacific Ocean is the largest ocean on Earth.",
        "metadata": {
            "context": "The Pacific Ocean is the largest and deepest of Earth's five oceans."
        },
    },
    {
        "id": "qa-006",
        "prompt": "How many continents are there on Earth?",
        "expected": "There are seven continents on Earth.",
        "metadata": {"context": "Earth is commonly divided into seven continents."},
    },
    {
        "id": "qa-007",
        "prompt": "What gas do plants absorb during photosynthesis?",
        "expected": "Plants absorb carbon dioxide during photosynthesis.",
        "metadata": {
            "context": "During photosynthesis, plants take in carbon dioxide and release oxygen."
        },
    },
    {
        "id": "qa-008",
        "prompt": "What is the chemical symbol for gold?",
        "expected": "The chemical symbol for gold is Au.",
        "metadata": {"context": "Gold's chemical symbol Au comes from the Latin word aurum."},
    },
    {
        "id": "qa-009",
        "prompt": "Which country is home to the kangaroo?",
        "expected": "Kangaroos are native to Australia.",
        "metadata": {"context": "The kangaroo is a marsupial indigenous to Australia."},
    },
    {
        "id": "qa-010",
        "prompt": "What is the smallest prime number?",
        "expected": "The smallest prime number is 2.",
        "metadata": {"context": "The number 2 is the smallest and the only even prime number."},
    },
    {
        "id": "qa-011",
        "prompt": "Who painted the Mona Lisa?",
        "expected": "The Mona Lisa was painted by Leonardo da Vinci.",
        "metadata": {
            "context": "Leonardo da Vinci painted the Mona Lisa in the early 16th century."
        },
    },
    {
        "id": "qa-012",
        "prompt": "What is the longest river in the world?",
        "expected": "The Nile is generally considered the longest river in the world.",
        "metadata": {
            "context": "The Nile in Africa is generally regarded as the longest river in the world."
        },
    },
    {
        "id": "qa-013",
        "prompt": "How many sides does a hexagon have?",
        "expected": "A hexagon has six sides.",
        "metadata": {"context": "A hexagon is a polygon with six sides and six angles."},
    },
    {
        "id": "qa-014",
        "prompt": "What is the freezing point of water in Fahrenheit?",
        "expected": "Water freezes at 32 degrees Fahrenheit.",
        "metadata": {"context": "On the Fahrenheit scale, water freezes at 32 degrees."},
    },
    {
        "id": "qa-015",
        "prompt": "Which element has the atomic number 1?",
        "expected": "Hydrogen has the atomic number 1.",
        "metadata": {
            "context": "Hydrogen is the first element of the periodic table with atomic number 1."
        },
    },
    {
        "id": "qa-016",
        "prompt": "What is the currency of Japan?",
        "expected": "The currency of Japan is the yen.",
        "metadata": {"context": "Japan's official currency is the Japanese yen."},
    },
    {
        "id": "qa-017",
        "prompt": "Who developed the theory of general relativity?",
        "expected": "Albert Einstein developed the theory of general relativity.",
        "metadata": {
            "context": "Albert Einstein published the theory of general relativity in 1915."
        },
    },
    {
        "id": "qa-018",
        "prompt": "What is the tallest mountain above sea level?",
        "expected": "Mount Everest is the tallest mountain above sea level.",
        "metadata": {"context": "Mount Everest is Earth's highest mountain above sea level."},
    },
    {
        "id": "qa-019",
        "prompt": "How many minutes are there in two hours?",
        "expected": "There are 120 minutes in two hours.",
        "metadata": {"context": "One hour has 60 minutes, so two hours contain 120 minutes."},
    },
    {
        "id": "qa-020",
        "prompt": "What is the primary language spoken in Brazil?",
        "expected": "Portuguese is the primary language spoken in Brazil.",
        "metadata": {"context": "Brazil's official and most widely spoken language is Portuguese."},
    },
    {
        "id": "qa-021",
        "prompt": "Which organ pumps blood through the human body?",
        "expected": "The heart pumps blood through the human body.",
        "metadata": {
            "context": "The heart is the muscular organ that circulates blood through the body."
        },
    },
    {
        "id": "qa-022",
        "prompt": "What is the square root of 144?",
        "expected": "The square root of 144 is 12.",
        "metadata": {"context": "Since 12 times 12 equals 144, the square root of 144 is 12."},
    },
    {
        "id": "qa-023",
        "prompt": "In which year did the Apollo 11 mission land humans on the Moon?",
        "expected": "Apollo 11 landed humans on the Moon in 1969.",
        "metadata": {
            "context": "The Apollo 11 mission achieved the first crewed Moon landing in July 1969."
        },
    },
    {
        "id": "qa-024",
        "prompt": "What is the capital of Japan?",
        "expected": "The capital of Japan is Tokyo.",
        "metadata": {"context": "Tokyo is the capital and most populous city of Japan."},
    },
    {
        "id": "qa-025",
        "prompt": "Which gas makes up most of Earth's atmosphere?",
        "expected": "Nitrogen makes up most of Earth's atmosphere.",
        "metadata": {"context": "Earth's atmosphere is about 78 percent nitrogen."},
    },
    {
        "id": "qa-026",
        "prompt": "Who wrote the novel Pride and Prejudice?",
        "expected": "Pride and Prejudice was written by Jane Austen.",
        "metadata": {"context": "Jane Austen published Pride and Prejudice in 1813."},
    },
    {
        "id": "qa-027",
        "prompt": "What is the largest planet in the solar system?",
        "expected": "Jupiter is the largest planet in the solar system.",
        "metadata": {
            "context": "Jupiter is the largest planet in our solar system by both mass and volume."
        },
    },
    {
        "id": "qa-028",
        "prompt": "How many bones does an adult human body have?",
        "expected": "An adult human body has 206 bones.",
        "metadata": {"context": "The adult human skeleton typically consists of 206 bones."},
    },
    {
        "id": "qa-029",
        "prompt": "What is the capital of Canada?",
        "expected": "The capital of Canada is Ottawa.",
        "metadata": {
            "context": "Ottawa, in the province of Ontario, is the capital city of Canada."
        },
    },
    {
        "id": "qa-030",
        "prompt": "Which metal is liquid at room temperature?",
        "expected": "Mercury is liquid at room temperature.",
        "metadata": {
            "context": "Mercury is the only metallic element that is liquid at room temperature."
        },
    },
    {
        "id": "qa-031",
        "prompt": "What does DNA stand for?",
        "expected": "DNA stands for deoxyribonucleic acid.",
        "metadata": {"context": "DNA, or deoxyribonucleic acid, carries genetic information."},
    },
    {
        "id": "qa-032",
        "prompt": "How many players are on a soccer team on the field?",
        "expected": "A soccer team has 11 players on the field.",
        "metadata": {"context": "Each soccer team fields 11 players, including the goalkeeper."},
    },
    {
        "id": "qa-033",
        "prompt": "What is the fastest land animal?",
        "expected": "The cheetah is the fastest land animal.",
        "metadata": {
            "context": "The cheetah can reach speeds around 100 kilometres per hour, making it the fastest land animal."
        },
    },
    {
        "id": "qa-034",
        "prompt": "Which ocean lies between Africa and Australia?",
        "expected": "The Indian Ocean lies between Africa and Australia.",
        "metadata": {
            "context": "The Indian Ocean is bounded by Africa to the west and Australia to the east."
        },
    },
    {
        "id": "qa-035",
        "prompt": "What is the chemical formula for table salt?",
        "expected": "The chemical formula for table salt is NaCl.",
        "metadata": {"context": "Table salt is sodium chloride, written as NaCl."},
    },
    {
        "id": "qa-036",
        "prompt": "Who was the first President of the United States?",
        "expected": "George Washington was the first President of the United States.",
        "metadata": {
            "context": "George Washington served as the first US president from 1789 to 1797."
        },
    },
    {
        "id": "qa-037",
        "prompt": "How many colors are there in a rainbow?",
        "expected": "A rainbow has seven colors.",
        "metadata": {"context": "A rainbow is commonly described as having seven colors."},
    },
    {
        "id": "qa-038",
        "prompt": "What is the hardest natural substance on Earth?",
        "expected": "Diamond is the hardest natural substance on Earth.",
        "metadata": {"context": "Diamond is the hardest known natural material."},
    },
    {
        "id": "qa-039",
        "prompt": "Which planet is closest to the Sun?",
        "expected": "Mercury is the planet closest to the Sun.",
        "metadata": {"context": "Mercury is the smallest planet and the nearest to the Sun."},
    },
    {
        "id": "qa-040",
        "prompt": "What is the sum of the interior angles of a triangle?",
        "expected": "The interior angles of a triangle sum to 180 degrees.",
        "metadata": {
            "context": "In Euclidean geometry, a triangle's interior angles always add up to 180 degrees."
        },
    },
]

_DEFAULT_EVALUATORS: list[EvaluatorConfig] = [
    ExactMatchConfig(strip_punctuation=True),
    TokenF1Config(),
    SemanticSimilarityConfig(),
    LLMJudgeConfig(
        rubric="correctness", provider=MockProviderConfig(model="sim-judge", quality=0.9, seed=7)
    ),
    SafetyConfig(blocked_terms=["confidential", "internal use only"]),
]

# (name, model, quality, seed) — a history of gradually improving nightly runs,
# then the pair used for the A/B comparison demo.
_DEMO_RUNS: list[tuple[str, str, float, int]] = [
    ("nightly-eval", "sim-small-v1", 0.55, 101),
    ("nightly-eval", "sim-small-v1", 0.60, 102),
    ("nightly-eval", "sim-small-v2", 0.66, 103),
    ("nightly-eval", "sim-small-v2", 0.70, 104),
    ("nightly-eval", "sim-small-v3", 0.74, 105),
    ("ab-baseline", "sim-small-v3", 0.72, 201),
    ("ab-candidate", "sim-large-v1", 0.88, 202),
]


def demo_config(dataset_path: str, name: str, model: str, quality: float, seed: int) -> EvalConfig:
    return EvalConfig(
        name=name,
        dataset=dataset_path,
        provider=MockProviderConfig(
            model=model,
            quality=quality,
            seed=seed,
            latency_ms=25.0,
            input_cost_per_1k_tokens=0.25,
            output_cost_per_1k_tokens=1.25,
        ),
        evaluators=list(_DEFAULT_EVALUATORS),
        prompt_template="Answer the question accurately and concisely.\n\nQuestion: {prompt}",
        concurrency=8,
        retries=1,
        retry_backoff_s=0.0,
    )


# Content-matched slice labels so subgroup analysis has meaningful groups.
_DEMO_TOPICS: dict[str, str] = {
    "qa-001": "geography",
    "qa-002": "science",
    "qa-003": "literature",
    "qa-004": "science",
    "qa-005": "geography",
    "qa-006": "geography",
    "qa-007": "science",
    "qa-008": "science",
    "qa-009": "geography",
    "qa-010": "math",
    "qa-011": "art",
    "qa-012": "geography",
    "qa-013": "math",
    "qa-014": "science",
    "qa-015": "science",
    "qa-016": "geography",
    "qa-017": "science",
    "qa-018": "geography",
    "qa-019": "math",
    "qa-020": "geography",
    "qa-021": "science",
    "qa-022": "math",
    "qa-023": "history",
    "qa-024": "geography",
    "qa-025": "science",
    "qa-026": "literature",
    "qa-027": "science",
    "qa-028": "science",
    "qa-029": "geography",
    "qa-030": "science",
    "qa-031": "science",
    "qa-032": "sports",
    "qa-033": "science",
    "qa-034": "geography",
    "qa-035": "science",
    "qa-036": "history",
    "qa-037": "science",
    "qa-038": "science",
    "qa-039": "science",
    "qa-040": "math",
}
_DEMO_DIFFICULTY: dict[str, str] = {
    "qa-007": "medium",
    "qa-008": "medium",
    "qa-010": "medium",
    "qa-012": "medium",
    "qa-014": "medium",
    "qa-015": "medium",
    "qa-017": "medium",
    "qa-023": "medium",
    "qa-025": "medium",
    "qa-026": "medium",
    "qa-029": "medium",
    "qa-030": "medium",
    "qa-031": "medium",
    "qa-034": "medium",
    "qa-035": "medium",
    "qa-038": "medium",
    "qa-040": "medium",
    "qa-028": "hard",
}


def write_demo_dataset(path: str) -> None:
    """Materialise the built-in demo items as a JSONL dataset file.

    Each item is tagged with ``topic`` and ``difficulty`` metadata so the
    subgroup (slice) analysis view has meaningful groups to break down.
    """
    import json
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for item in DEMO_ITEMS:
            enriched = dict(item)
            raw_metadata = enriched.get("metadata", {})
            metadata: dict[str, str] = {}
            if isinstance(raw_metadata, dict):
                metadata.update({str(k): str(v) for k, v in raw_metadata.items()})
            metadata["topic"] = _DEMO_TOPICS.get(str(item["id"]), "general")
            metadata["difficulty"] = _DEMO_DIFFICULTY.get(str(item["id"]), "easy")
            enriched["metadata"] = metadata
            handle.write(json.dumps(enriched) + "\n")


async def seed_demo(storage: Storage, dataset_path: str) -> list[str]:
    """Run the demo evaluation history; returns the created run ids in order."""
    write_demo_dataset(dataset_path)
    run_ids: list[str] = []
    for name, model, quality, seed in _DEMO_RUNS:
        config = demo_config(dataset_path, name, model, quality, seed)
        result = await execute_run(config, storage)
        run_ids.append(result.run_id)
    storage.save_prompt(
        "concise-qa",
        "Answer the question accurately and concisely.\n\nQuestion: {prompt}",
    )
    return run_ids
