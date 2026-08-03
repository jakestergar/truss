from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from truss.base.constants import (
    OPENAI_COMPATIBLE_TAG,
    OPENAI_NON_COMPATIBLE_TAG,
    TRTLLM_MIN_MEMORY_REQUEST_GI,
)
from truss.base.trt_llm_config import TrussTRTLLMModel
from truss.trt_llm.config_checks import (
    _is_model_public,
    has_no_tags_trt_llm_builder,
    memory_updated_for_trt_llm_builder,
    uses_trt_llm_builder,
)


def make_truss(
    *,
    trt_llm=True,
    inference_stack="v1",
    base_model="decoder",
    speculator=None,
    tags=None,
    memory_in_bytes=100 * 1024**3,
):
    config = SimpleNamespace(
        trt_llm=(
            SimpleNamespace(
                root=SimpleNamespace(
                    inference_stack=inference_stack,
                    build=SimpleNamespace(base_model=base_model, speculator=speculator),
                )
            )
            if trt_llm
            else None
        ),
        model_metadata={"tags": tags} if tags is not None else {},
        resources=SimpleNamespace(memory=None),
        config_path="/tmp/config.yaml",
    )
    return SimpleNamespace(
        spec=SimpleNamespace(
            config=config,
            memory_in_bytes=memory_in_bytes,
            config_path="/tmp/config.yaml",
        ),
        config_path="/tmp/config.yaml",
    )


@pytest.mark.parametrize(
    "truss",
    [
        make_truss(trt_llm=False),
        make_truss(inference_stack="v2"),
        make_truss(base_model=TrussTRTLLMModel.ENCODER),
        make_truss(base_model=TrussTRTLLMModel.ENCODER_BERT),
    ],
)
def test_tag_check_skips_non_briton_models(truss):
    assert has_no_tags_trt_llm_builder(truss) == ("", False)


def test_tag_check_adds_openai_tag_for_speculator():
    truss = make_truss(speculator="draft", tags=[])
    truss.spec.config.write_to_yaml_file = Mock()
    with patch.object(
        truss.spec.config,
        "write_to_yaml_file",
        wraps=truss.spec.config.write_to_yaml_file,
    ) as write:
        message, is_error = has_no_tags_trt_llm_builder(truss)
    assert not is_error
    assert OPENAI_COMPATIBLE_TAG in truss.spec.config.model_metadata["tags"]
    assert "openai-compatible tag" in message
    write.assert_called_once_with("/tmp/config.yaml", verbose=False)


def test_tag_check_rejects_non_openai_speculator():
    truss = make_truss(speculator="draft", tags=[OPENAI_NON_COMPATIBLE_TAG])
    message, is_error = has_no_tags_trt_llm_builder(truss)
    assert is_error
    assert OPENAI_NON_COMPATIBLE_TAG in message


def test_tag_check_adds_tag_when_no_tag_exists():
    truss = make_truss(tags=[])
    truss.spec.config.write_to_yaml_file = Mock()
    with patch.object(truss.spec.config, "write_to_yaml_file"):
        message, is_error = has_no_tags_trt_llm_builder(truss)
    assert not is_error
    assert OPENAI_COMPATIBLE_TAG in truss.spec.config.model_metadata["tags"]
    assert "require a model_metadata" in message


@pytest.mark.parametrize("tags", [[OPENAI_COMPATIBLE_TAG], [OPENAI_NON_COMPATIBLE_TAG]])
def test_tag_check_handles_existing_tags(tags):
    truss = make_truss(tags=tags)
    message, is_error = has_no_tags_trt_llm_builder(truss)
    assert not is_error
    if tags == [OPENAI_NON_COMPATIBLE_TAG]:
        assert "deprecated" in message
    else:
        assert message == ""


def test_memory_check_updates_small_request():
    truss = make_truss(memory_in_bytes=1)
    truss.spec.config.write_to_yaml_file = Mock()
    with patch.object(
        truss.spec.config,
        "write_to_yaml_file",
        wraps=truss.spec.config.write_to_yaml_file,
    ) as write:
        assert memory_updated_for_trt_llm_builder(truss)
    assert truss.spec.config.resources.memory == f"{TRTLLM_MIN_MEMORY_REQUEST_GI}Gi"
    write.assert_called_once_with("/tmp/config.yaml", verbose=False)


def test_memory_check_does_not_update_non_trt_or_sufficient_memory():
    assert not memory_updated_for_trt_llm_builder(
        make_truss(trt_llm=False, memory_in_bytes=1)
    )
    assert not memory_updated_for_trt_llm_builder(make_truss())


@pytest.mark.parametrize("status, expected", [(200, True), (401, False)])
def test_is_model_public(status, expected):
    response = Mock(status_code=status)
    with patch(
        "truss.trt_llm.config_checks.requests.get", return_value=response
    ) as get:
        assert _is_model_public("org/model") is expected
    get.assert_called_once()


def test_uses_trt_llm_builder():
    assert uses_trt_llm_builder(make_truss())
    assert not uses_trt_llm_builder(make_truss(trt_llm=False))
