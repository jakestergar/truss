import enum
import logging

import pydantic
import pytest

from truss_chains import public_types
from truss_chains.utils import (
    InjectedError,
    StrEnum,
    expect_one,
    get_pydantic_field_default_value,
    issubclass_safe,
    log_level,
    make_abs_path_here,
    make_optional_import_error,
    random_fail,
    setup_dev_logging,
)


def test_make_abs_path_here(tmp_path):
    existing = tmp_path / "existing"
    existing.write_text("ok")
    result = make_abs_path_here(str(existing))
    assert result.abs_path == str(existing)


def test_setup_dev_logging_with_handlers():
    logger = logging.getLogger()
    original = list(logger.handlers)
    handler = logging.StreamHandler()
    logger.handlers = [handler]
    setup_dev_logging(logging.DEBUG)
    assert logger.level == logging.DEBUG
    assert logger.handlers == [handler]
    assert handler.formatter is not None
    logger.handlers = original


def test_setup_dev_logging_without_handlers():
    logger = logging.getLogger()
    original = list(logger.handlers)
    logger.handlers = []
    setup_dev_logging()
    assert logger.handlers
    logger.handlers = original


def test_log_level_context():
    logger = logging.getLogger()
    original = logger.level
    logger.setLevel(logging.INFO)
    with log_level(logging.DEBUG):
        assert logger.level == logging.DEBUG
    assert logger.level == logging.INFO
    logger.setLevel(original)


def test_expect_one_single():
    assert expect_one([42]) == 42


def test_expect_one_empty():
    with pytest.raises(ValueError, match="empty"):
        expect_one([])


def test_expect_one_multiple():
    with pytest.raises(ValueError, match="more than one"):
        expect_one([1, 2])


def test_random_fail_certain():
    with pytest.raises(InjectedError, match="boom"):
        random_fail(1.0, "boom")


def test_random_fail_never():
    random_fail(0.0, "boom")


def test_str_enum_auto():
    class Color(StrEnum):
        RED = enum.auto()
        BLUE = enum.auto()

    assert Color.RED == "RED"
    assert Color("RED") == Color.RED


def test_str_enum_invalid_value():
    with pytest.raises(TypeError):

        class Bad(StrEnum):
            ONE = 1


def test_str_enum_lowercase_rejected():
    with pytest.raises(ValueError):

        class Bad(StrEnum):
            lower = enum.auto()


def test_issubclass_safe():
    assert issubclass_safe(int, object) is True
    assert issubclass_safe(1, object) is False


def test_get_pydantic_field_default_value():
    class M(pydantic.BaseModel):
        a: int = 1
        b: list = pydantic.Field(default_factory=list)
        c: str

    assert get_pydantic_field_default_value(M, "a") == 1
    assert get_pydantic_field_default_value(M, "b") == []
    assert get_pydantic_field_default_value(M, "c") is None


def test_make_optional_import_error():
    err = make_optional_import_error("some_module")
    assert isinstance(err, public_types.ChainsRuntimeError)
    assert "some_module" in str(err)
