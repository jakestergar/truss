from unittest.mock import Mock

from truss.base.constants import (
    OPENAI_COMPATIBLE_TAG,
    OPENAI_NON_COMPATIBLE_TAG,
    TRTLLM_MIN_MEMORY_REQUEST_GI,
)
from truss.base.trt_llm_config import TrussTRTLLMModel
from truss.trt_llm.config_checks import (
    has_no_tags_trt_llm_builder,
    memory_updated_for_trt_llm_builder,
    uses_trt_llm_builder,
)


def _make_truss_handle(memory_in_bytes=4 * 1024**3, inference_stack="v1"):
    tr = Mock()
    tr.spec.config.trt_llm.root.build.base_model = TrussTRTLLMModel.LLAMA
    tr.spec.config.trt_llm.root.build.speculator = None
    tr.spec.config.trt_llm.root.inference_stack = inference_stack
    tr.spec.config.model_metadata = {}
    tr.spec.memory_in_bytes = memory_in_bytes
    tr.spec.config.resources.memory = "2Gi"
    tr.spec.config_path = "/tmp/config.yaml"
    return tr


def test_uses_trt_llm_builder():
    assert uses_trt_llm_builder(_make_truss_handle()) is True
    tr = Mock()
    tr.spec.config.trt_llm = None
    assert uses_trt_llm_builder(tr) is False


def test_has_no_tags_adds_openai_tag():
    tr = _make_truss_handle()
    message, should_block = has_no_tags_trt_llm_builder(tr)
    assert not should_block
    assert OPENAI_COMPATIBLE_TAG in message
    assert tr.spec.config.model_metadata["tags"] == [OPENAI_COMPATIBLE_TAG]
    tr.spec.config.write_to_yaml_file.assert_called_once_with(
        tr.spec.config_path, verbose=False
    )


def test_has_no_tags_with_non_compatible_tag():
    tr = _make_truss_handle()
    tr.spec.config.model_metadata["tags"] = [OPENAI_NON_COMPATIBLE_TAG]
    message, should_block = has_no_tags_trt_llm_builder(tr)
    assert not should_block
    assert "deprecated" in message


def test_has_no_tags_with_compatible_tag():
    tr = _make_truss_handle()
    tr.spec.config.model_metadata["tags"] = [OPENAI_COMPATIBLE_TAG]
    assert has_no_tags_trt_llm_builder(tr) == ("", False)


def test_has_no_tags_v2_stack():
    tr = _make_truss_handle(inference_stack="v2")
    assert has_no_tags_trt_llm_builder(tr) == ("", False)


def test_has_no_tags_encoder():
    tr = _make_truss_handle()
    tr.spec.config.trt_llm.root.build.base_model = TrussTRTLLMModel.ENCODER
    assert has_no_tags_trt_llm_builder(tr) == ("", False)


def test_memory_updated_for_trt_llm_builder_low_memory():
    tr = _make_truss_handle(memory_in_bytes=1 * 1024**3)
    assert memory_updated_for_trt_llm_builder(tr) is True
    assert tr.spec.config.resources.memory == f"{TRTLLM_MIN_MEMORY_REQUEST_GI}Gi"


def test_memory_updated_for_trt_llm_builder_sufficient_memory():
    tr = _make_truss_handle(memory_in_bytes=100 * 1024**3)
    assert memory_updated_for_trt_llm_builder(tr) is False
