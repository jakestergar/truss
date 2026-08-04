import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

from truss.base.constants import (
    CONFIG_FILE,
    HF_MODELS_API_URL,
    OPENAI_COMPATIBLE_TAG,
    OPENAI_NON_COMPATIBLE_TAG,
    TRTLLM_MIN_MEMORY_REQUEST_GI,
)
from truss.trt_llm.config_checks import (
    _is_model_public,
    has_no_tags_trt_llm_builder,
    memory_updated_for_trt_llm_builder,
    uses_trt_llm_builder,
)
from truss.truss_handle.truss_handle import TrussHandle


@pytest.fixture
def truss_dir(custom_model_trt_llm) -> Path:
    """Truss dir whose model class accepts the `trt_llm` init arg (required by validation)."""
    return custom_model_trt_llm


def _make_handle(
    truss_dir, config: Dict[str, Any], tags: Optional[List[str]] = None
) -> TrussHandle:
    """Write `config` (plus optional model_metadata tags) into an existing truss dir."""
    config = copy.deepcopy(config)
    if tags is not None:
        config["model_metadata"] = {"tags": tags}
    (truss_dir / CONFIG_FILE).write_text(yaml.safe_dump(config))
    return TrussHandle(truss_dir)


def _written_tags(handle: TrussHandle) -> List[str]:
    return yaml.safe_load(handle.spec.config_path.read_text())["model_metadata"]["tags"]


def test_uses_trt_llm_builder_without_trt_llm(truss_dir, default_config):
    handle = _make_handle(truss_dir, default_config)
    assert not uses_trt_llm_builder(handle)


def test_uses_trt_llm_builder_with_trt_llm(truss_dir, trtllm_config):
    handle = _make_handle(truss_dir, trtllm_config)
    assert uses_trt_llm_builder(handle)


def test_memory_not_updated_without_trt_llm(truss_dir, default_config):
    handle = _make_handle(truss_dir, default_config)
    assert not memory_updated_for_trt_llm_builder(handle)
    assert handle.spec.memory == "2Gi"


def test_memory_not_updated_when_already_sufficient(truss_dir, trtllm_config):
    handle = _make_handle(truss_dir, trtllm_config)
    assert not memory_updated_for_trt_llm_builder(handle)
    assert handle.spec.memory == "24Gi"


def test_memory_updated_when_below_minimum(truss_dir, trtllm_config):
    trtllm_config["resources"]["memory"] = "2Gi"
    handle = _make_handle(truss_dir, trtllm_config)

    assert memory_updated_for_trt_llm_builder(handle)

    expected = f"{TRTLLM_MIN_MEMORY_REQUEST_GI}Gi"
    assert handle.spec.memory == expected
    assert handle.spec.memory_in_bytes == TRTLLM_MIN_MEMORY_REQUEST_GI * 1024**3
    # the updated memory is persisted to config.yaml.
    written = yaml.safe_load(handle.spec.config_path.read_text())
    assert written["resources"]["memory"] == expected


def test_has_no_tags_without_trt_llm(truss_dir, default_config):
    handle = _make_handle(truss_dir, default_config)
    assert has_no_tags_trt_llm_builder(handle) == ("", False)


def test_has_no_tags_ignores_inference_stack_v2(truss_dir, trtllm_config_v2):
    handle = _make_handle(truss_dir, trtllm_config_v2)
    assert has_no_tags_trt_llm_builder(handle) == ("", False)
    assert handle.spec.config.model_metadata.get("tags") is None


@pytest.mark.parametrize("base_model", ["encoder", "encoder_bert"])
def test_has_no_tags_ignores_encoder_models(
    truss_dir, trtllm_config_encoder, base_model
):
    trtllm_config_encoder["trt_llm"]["build"]["base_model"] = base_model
    handle = _make_handle(truss_dir, trtllm_config_encoder)
    assert has_no_tags_trt_llm_builder(handle) == ("", False)


@pytest.mark.parametrize("existing_tags", [None, [], ["some-other-tag"]])
def test_has_no_tags_adds_openai_tag_when_missing(
    truss_dir, trtllm_config, existing_tags
):
    handle = _make_handle(truss_dir, trtllm_config, tags=existing_tags)

    message, should_raise = has_no_tags_trt_llm_builder(handle)

    assert not should_raise
    assert OPENAI_COMPATIBLE_TAG in message
    assert OPENAI_NON_COMPATIBLE_TAG in message
    # the tag is prepended in memory and persisted to config.yaml.
    expected_tags = [OPENAI_COMPATIBLE_TAG] + (existing_tags or [])
    assert handle.spec.config.model_metadata["tags"] == expected_tags
    assert _written_tags(handle) == expected_tags


def test_has_no_tags_accepts_openai_compatible_tag(truss_dir, trtllm_config):
    handle = _make_handle(truss_dir, trtllm_config, tags=[OPENAI_COMPATIBLE_TAG])
    assert has_no_tags_trt_llm_builder(handle) == ("", False)
    assert handle.spec.config.model_metadata["tags"] == [OPENAI_COMPATIBLE_TAG]


def test_has_no_tags_warns_on_deprecated_non_compatible_tag(truss_dir, trtllm_config):
    handle = _make_handle(truss_dir, trtllm_config, tags=[OPENAI_NON_COMPATIBLE_TAG])

    message, should_raise = has_no_tags_trt_llm_builder(handle)

    assert not should_raise
    assert "deprecated" in message
    assert OPENAI_NON_COMPATIBLE_TAG in message
    # the deprecated tag is left untouched.
    assert handle.spec.config.model_metadata["tags"] == [OPENAI_NON_COMPATIBLE_TAG]


def test_has_no_tags_speculator_rejects_non_compatible_tag(
    truss_dir, trtllm_spec_dec_config_lookahead_v1
):
    handle = _make_handle(
        truss_dir, trtllm_spec_dec_config_lookahead_v1, tags=[OPENAI_NON_COMPATIBLE_TAG]
    )

    message, should_raise = has_no_tags_trt_llm_builder(handle)

    assert should_raise
    assert "speculator" in message
    assert OPENAI_NON_COMPATIBLE_TAG in message
    assert handle.spec.config.model_metadata["tags"] == [OPENAI_NON_COMPATIBLE_TAG]


def test_has_no_tags_speculator_adds_openai_tag_when_missing(
    truss_dir, trtllm_spec_dec_config_lookahead_v1
):
    handle = _make_handle(truss_dir, trtllm_spec_dec_config_lookahead_v1)

    message, should_raise = has_no_tags_trt_llm_builder(handle)

    assert not should_raise
    assert "speculator" in message
    assert OPENAI_COMPATIBLE_TAG in message
    assert handle.spec.config.model_metadata["tags"] == [OPENAI_COMPATIBLE_TAG]
    assert _written_tags(handle) == [OPENAI_COMPATIBLE_TAG]


def test_has_no_tags_speculator_with_openai_tag(
    truss_dir, trtllm_spec_dec_config_lookahead_v1
):
    handle = _make_handle(
        truss_dir, trtllm_spec_dec_config_lookahead_v1, tags=[OPENAI_COMPATIBLE_TAG]
    )
    assert has_no_tags_trt_llm_builder(handle) == ("", False)


@pytest.mark.parametrize(
    "status_code, expected", [(200, True), (401, False), (404, False), (500, False)]
)
def test_is_model_public(requests_mock, status_code, expected):
    model_id = "meta/llama4-500B"
    requests_mock.get(f"{HF_MODELS_API_URL}/{model_id}", status_code=status_code)

    assert _is_model_public(model_id) is expected
