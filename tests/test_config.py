"""Config validation: discriminated unions, bounds, template rules, YAML errors."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from evalpipe.config import (
    EvalConfig,
    LLMJudgeConfig,
    MockProviderConfig,
    OpenAICompatibleProviderConfig,
    RegexConfig,
    load_config,
)
from evalpipe.exceptions import ConfigError

VALID_YAML = """
name: demo
dataset: data.jsonl
provider:
  type: mock
  quality: 0.9
evaluators:
  - type: exact_match
  - type: token_f1
    threshold: 0.5
"""


def test_load_valid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(VALID_YAML, encoding="utf-8")
    config = load_config(path)
    assert config.name == "demo"
    assert isinstance(config.provider, MockProviderConfig)
    assert config.provider.quality == 0.9
    assert len(config.evaluators) == 2


def test_missing_file() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("does-not-exist.yaml")


def test_invalid_yaml_syntax(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("name: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_non_mapping_top_level(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_unknown_evaluator_type_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(VALID_YAML.replace("token_f1", "made_up_metric"), encoding="utf-8")
    with pytest.raises(ConfigError, match="evaluators"):
        load_config(path)


def test_empty_evaluators_rejected() -> None:
    with pytest.raises(ValidationError):
        EvalConfig(name="x", dataset="d.jsonl", provider=MockProviderConfig(), evaluators=[])


def test_quality_bounds() -> None:
    with pytest.raises(ValidationError):
        MockProviderConfig(quality=1.5)
    with pytest.raises(ValidationError):
        MockProviderConfig(quality=-0.1)


def test_base_url_scheme_required() -> None:
    with pytest.raises(ValidationError, match="base_url"):
        OpenAICompatibleProviderConfig(model="m", base_url="localhost:8080")


def test_base_url_trailing_slash_stripped() -> None:
    config = OpenAICompatibleProviderConfig(model="m", base_url="http://host/v1/")
    assert config.base_url == "http://host/v1"


def test_invalid_regex_rejected() -> None:
    with pytest.raises(ValidationError, match="regular expression"):
        RegexConfig(pattern="[unclosed")


def test_prompt_template_requires_placeholder() -> None:
    with pytest.raises(ValidationError, match="placeholder"):
        EvalConfig(
            name="x",
            dataset="d.jsonl",
            provider=MockProviderConfig(),
            evaluators=[LLMJudgeConfig()],
            prompt_template="no placeholder here",
        )


def test_render_prompt() -> None:
    config = EvalConfig(
        name="x",
        dataset="d.jsonl",
        provider=MockProviderConfig(),
        evaluators=[LLMJudgeConfig()],
        prompt_template="Context first.\n\nQ: {prompt}",
    )
    assert config.render_prompt("Why?") == "Context first.\n\nQ: Why?"
    config_plain = config.model_copy(update={"prompt_template": None})
    assert config_plain.render_prompt("Why?") == "Why?"


def test_extra_keys_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(VALID_YAML + "\ntypo_key: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="typo_key"):
        load_config(path)
