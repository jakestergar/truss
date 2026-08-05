import datetime
import pickle
import uuid
from decimal import Decimal

import msgpack
import numpy as np
import pytest

from truss.templates.shared import serialization


class _Exploit:
    def __reduce__(self):
        return (print, ("pwned",))


def test_roundtrip_basic_types():
    obj = {
        "str": "a",
        "int": 1,
        "float": 1.5,
        "bool": True,
        "none": None,
        "list": [1, 2],
        "datetime": datetime.datetime(2024, 1, 1, 12, 0),
        "date": datetime.date(2024, 1, 1),
        "time": datetime.time(12, 0),
        "timedelta": datetime.timedelta(days=1, seconds=2, microseconds=3),
        "decimal": Decimal("1.5"),
        "uuid": uuid.uuid4(),
    }
    assert (
        serialization.truss_msgpack_deserialize(
            serialization.truss_msgpack_serialize(obj)
        )
        == obj
    )


@pytest.mark.parametrize(
    "array",
    [
        np.arange(6, dtype=np.float32).reshape(2, 3),
        np.array([True, False]),
        np.array([(1, 2.0)], dtype=[("a", "i4"), ("b", "f8")]),
    ],
)
def test_roundtrip_numpy(array):
    result = serialization.truss_msgpack_deserialize(
        serialization.truss_msgpack_serialize({"a": array})
    )
    np.testing.assert_array_equal(result["a"], array)


def test_deserialize_rejects_pickle_payload():
    payload = msgpack.packb(
        {
            b"nd": True,
            b"kind": b"O",
            b"type": "|O",
            b"shape": (1,),
            b"data": pickle.dumps(_Exploit()),
        }
    )
    with pytest.raises(ValueError, match="object-dtype"):
        serialization.truss_msgpack_deserialize(payload)


def test_serialize_rejects_object_array():
    with pytest.raises(ValueError, match="object-dtype"):
        serialization.truss_msgpack_serialize({"a": np.array([_Exploit()])})
