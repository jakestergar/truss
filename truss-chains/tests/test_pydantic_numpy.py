import base64

import numpy as np
import pydantic
import pytest

from truss_chains.pydantic_numpy import NumpyArrayField


class _ArrayModel(pydantic.BaseModel):
    array: NumpyArrayField


def test_numpy_array_field_roundtrip():
    arr = np.arange(4).reshape((2, 2))
    model = _ArrayModel(array=arr)
    json_str = model.model_dump_json()
    restored = _ArrayModel.model_validate_json(json_str)
    np.testing.assert_array_equal(restored.array.array, arr)


def test_numpy_array_field_from_dict():
    arr = np.array([1.0, 2.0, 3.0])
    encoded = base64.b64encode(arr.tobytes()).decode("utf-8")
    model = _ArrayModel.model_validate(
        {
            "array": {
                NumpyArrayField.data_key: encoded,
                NumpyArrayField.shape_key: list(arr.shape),
                NumpyArrayField.dtype_key: str(arr.dtype),
            }
        }
    )
    np.testing.assert_array_equal(model.array.array, arr)


def test_numpy_array_field_repr():
    arr = np.array([1, 2, 3])
    field = NumpyArrayField(arr)
    assert "NumpyArrayField" in repr(field)
    assert "(3,)" in repr(field)


def test_numpy_array_field_invalid_value():
    with pytest.raises(TypeError, match="Expected a NumPy array"):
        _ArrayModel(array="not-an-array")


def test_numpy_array_field_json_schema():
    schema = _ArrayModel.model_json_schema()
    assert "array" in schema["properties"]
    array_schema = schema["properties"]["array"]
    assert array_schema["type"] == "object"
    assert "data" in array_schema["properties"]
    assert "shape" in array_schema["properties"]
    assert "dtype" in array_schema["properties"]
