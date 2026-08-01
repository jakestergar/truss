import base64

import numpy as np
import pydantic
import pytest

from truss_chains import pydantic_numpy


class _Model(pydantic.BaseModel):
    array: pydantic_numpy.NumpyArrayField


def test_accepts_numpy_array_and_roundtrips_json():
    model = _Model(array=np.arange(4).reshape((2, 2)))
    restored = _Model.model_validate_json(model.model_dump_json())
    np.testing.assert_array_equal(restored.array.array, model.array.array)
    assert restored.array.array.dtype == model.array.array.dtype


def test_json_dump_shape_and_dtype():
    array = np.arange(4, dtype=np.int32).reshape((2, 2))
    dumped = _Model(array=array).model_dump(mode="json")["array"]
    assert list(dumped["shape"]) == [2, 2]
    assert dumped["dtype"] == "int32"
    assert base64.b64decode(dumped["data_b64"]) == array.tobytes()


def test_python_mode_dump_returns_raw_array():
    array = np.arange(3)
    dumped = _Model(array=array).model_dump()["array"]
    assert isinstance(dumped, np.ndarray)
    np.testing.assert_array_equal(dumped, array)


def test_accepts_existing_field_instance():
    field = pydantic_numpy.NumpyArrayField(np.arange(2))
    assert _Model(array=field).array is field


def test_validate_from_dict():
    array = np.arange(4, dtype=np.float64)
    value = {
        "data_b64": base64.b64encode(array.tobytes()).decode("utf-8"),
        "shape": [4],
        "dtype": "float64",
    }
    np.testing.assert_array_equal(
        pydantic_numpy.NumpyArrayField.validate_numpy_array(value).array, array
    )


def test_invalid_dict_payload_raises_type_error():
    value = {"data_b64": "AAAA", "shape": [7, 7], "dtype": "float64"}
    with pytest.raises(TypeError, match="Invalid data, shape, or dtype"):
        pydantic_numpy.NumpyArrayField.validate_numpy_array(value)


@pytest.mark.parametrize("value", ["not-an-array", 3, {"shape": [1]}, None])
def test_unsupported_value_raises_type_error(value):
    with pytest.raises(TypeError, match="Expected a NumPy array"):
        pydantic_numpy.NumpyArrayField.validate_numpy_array(value)


def test_repr_includes_shape_and_dtype():
    text = repr(pydantic_numpy.NumpyArrayField(np.zeros((2, 3), dtype=np.int8)))
    assert "shape=(2, 3)" in text
    assert "dtype=int8" in text


def test_json_schema_describes_serialized_form():
    schema = _Model.model_json_schema()["properties"]["array"]
    assert schema["type"] == "object"
    assert schema["required"] == ["data", "shape", "dtype"]
    assert schema["properties"]["data"] == {"type": "string", "format": "byte"}
