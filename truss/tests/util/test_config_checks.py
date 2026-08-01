from unittest.mock import patch

import pytest

from truss.base.constants import (
    HF_MODELS_API_URL,
    OPENAI_COMPATIBLE_TAG,
    OPENAI_NON_COMPATIBLE_TAG,
    TRTLLM_MIN_MEMORY_REQUEST_GI,
)
from truss.base.trt_llm_config import (
    TrussSpeculatorConfiguration,
    TrussTRTLLMBuildConfiguration,
    TrussTRTLLMModel,
)
from truss.trt_llm.config_checks import (
    _is_model_public,
    has_no_tags_trt_llm_builder,
    memory_updated_for_trt_llm_builder,
    uses_trt_llm_builder,
)
from truss.truss_handle.truss_handle import TrussHandle


def test_check_and_update_memory_for_trt_llm_builder(custom_model_trt_llm):
    handle = TrussHandle(custom_model_trt_llm)
    assert memory_updated_for_trt_llm_builder(handle)
    assert handle.spec.memory == f"{TRTLLM_MIN_MEMORY_REQUEST_GI}Gi"
    assert handle.spec.memory_in_bytes == TRTLLM_MIN_MEMORY_REQUEST_GI * 1024**3


def test_config_checks_return_defaults_for_non_trt_llm(custom_model_trt_llm):
    handle = TrussHandle(custom_model_trt_llm)
    handle.spec.config.trt_llm = None

    assert not uses_trt_llm_builder(handle)
    assert has_no_tags_trt_llm_builder(handle) == ("", False)
    assert not memory_updated_for_trt_llm_builder(handle)


@pytest.mark.parametrize(
    "base_model", [TrussTRTLLMModel.ENCODER, TrussTRTLLMModel.ENCODER_BERT]
)
def test_has_no_tags_skips_encoder_models(custom_model_trt_llm, base_model):
    handle = TrussHandle(custom_model_trt_llm)
    handle.spec.config.trt_llm.root.build.base_model = base_model

    assert has_no_tags_trt_llm_builder(handle) == ("", False)


def test_has_no_tags_skips_inference_stack_v2(custom_model_trt_llm_stack_v2):
    handle = TrussHandle(custom_model_trt_llm_stack_v2)

    assert has_no_tags_trt_llm_builder(handle) == ("", False)


def test_has_no_tags_adds_compatibility_tag_for_plain_briton(custom_model_trt_llm):
    handle = TrussHandle(custom_model_trt_llm)

    message, is_error = has_no_tags_trt_llm_builder(handle)

    assert not is_error
    assert OPENAI_COMPATIBLE_TAG in message
    assert handle.spec.config.model_metadata["tags"] == [OPENAI_COMPATIBLE_TAG]


def test_has_no_tags_warns_for_deprecated_tag(custom_model_trt_llm):
    handle = TrussHandle(custom_model_trt_llm)
    handle.spec.config.model_metadata["tags"] = [OPENAI_NON_COMPATIBLE_TAG]

    message, is_error = has_no_tags_trt_llm_builder(handle)

    assert not is_error
    assert "deprecated" in message
    assert OPENAI_NON_COMPATIBLE_TAG in message


def test_has_no_tags_accepts_compatible_tag(custom_model_trt_llm):
    handle = TrussHandle(custom_model_trt_llm)
    handle.spec.config.model_metadata["tags"] = [OPENAI_COMPATIBLE_TAG]

    assert has_no_tags_trt_llm_builder(handle) == ("", False)


def test_has_no_tags_speculator_rejects_non_compatible_tag(custom_model_trt_llm):
    handle = TrussHandle(custom_model_trt_llm)
    handle.spec.config.trt_llm.root.build.speculator = TrussSpeculatorConfiguration(
        num_draft_tokens=1, build=TrussTRTLLMBuildConfiguration()
    )
    handle.spec.config.model_metadata["tags"] = [OPENAI_NON_COMPATIBLE_TAG]

    message, is_error = has_no_tags_trt_llm_builder(handle)

    assert is_error
    assert OPENAI_NON_COMPATIBLE_TAG in message
    assert OPENAI_COMPATIBLE_TAG in message


def test_has_no_tags_speculator_adds_compatible_tag(custom_model_trt_llm):
    handle = TrussHandle(custom_model_trt_llm)
    handle.spec.config.trt_llm.root.build.speculator = TrussSpeculatorConfiguration(
        num_draft_tokens=1, build=TrussTRTLLMBuildConfiguration()
    )

    message, is_error = has_no_tags_trt_llm_builder(handle)

    assert not is_error
    assert OPENAI_COMPATIBLE_TAG in message
    assert handle.spec.config.model_metadata["tags"] == [OPENAI_COMPATIBLE_TAG]


def test_memory_is_not_updated_when_already_above_threshold(custom_model_trt_llm):
    handle = TrussHandle(custom_model_trt_llm)
    handle.spec.config.resources.memory = f"{TRTLLM_MIN_MEMORY_REQUEST_GI + 1}Gi"

    assert not memory_updated_for_trt_llm_builder(handle)
    assert handle.spec.memory == f"{TRTLLM_MIN_MEMORY_REQUEST_GI + 1}Gi"


@pytest.mark.parametrize(("status_code", "expected"), [(200, True), (401, False)])
def test_is_model_public(status_code, expected):
    response = type("Response", (), {"status_code": status_code})()
    with patch(
        "truss.trt_llm.config_checks.requests.get", return_value=response
    ) as get:
        assert _is_model_public("google/gemma") is expected

    get.assert_called_once_with(f"{HF_MODELS_API_URL}/google/gemma")
