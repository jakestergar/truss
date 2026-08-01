import json
import uuid
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest

from truss.templates.shared import serialization


@pytest.mark.parametrize(
    "value",
    [
        datetime(2024, 1, 2, 3, 4, 5),
        datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        date(2024, 1, 2),
        time(3, 4, 5),
        timedelta(days=1, seconds=2, microseconds=3),
        Decimal("1.25"),
        uuid.UUID("12345678-1234-5678-1234-567812345678"),
    ],
)
def test_msgpack_roundtrip_of_extended_types(value):
    data = serialization.truss_msgpack_serialize({"v": value})
    assert serialization.truss_msgpack_deserialize(data)["v"] == value


def test_utc_datetime_is_encoded_with_z_suffix():
    encoded = serialization._truss_msgpack_encoder(
        datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    )
    assert encoded[b"data"].endswith("Z")


def test_timezone_aware_time_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware times"):
        serialization._truss_msgpack_encoder(time(3, 4, 5, tzinfo=timezone.utc))


def test_encoder_passes_unknown_object_to_chain():
    sentinel = object()
    assert serialization._truss_msgpack_encoder(sentinel) is sentinel
    assert serialization._truss_msgpack_encoder(
        sentinel, chain=lambda o: "chained"
    ) == ("chained")


def test_decoder_passes_plain_mapping_through():
    obj = {b"other": 1}
    assert serialization._truss_msgpack_decoder(obj) is obj
    assert serialization._truss_msgpack_decoder(obj, chain=lambda o: "chained") == (
        "chained"
    )


def test_decoder_falls_back_to_chain_on_key_error():
    class _NoData(dict):
        def __getitem__(self, key):
            raise KeyError(key)

    obj = _NoData({b"__decimal__": True})
    assert serialization._truss_msgpack_decoder(obj) is obj
    assert serialization._truss_msgpack_decoder(obj, chain=lambda o: "chained") == (
        "chained"
    )


def test_msgpack_roundtrip_of_numpy_array():
    array = np.arange(6, dtype=np.float32).reshape((2, 3))
    data = serialization.truss_msgpack_serialize({"a": array})
    result = serialization.truss_msgpack_deserialize(data)["a"]
    np.testing.assert_array_equal(result, array)


def test_msgpack_roundtrip_of_nested_json_types():
    payload = {"s": "x", "i": 1, "f": 1.5, "b": True, "n": None, "l": [1, 2]}
    assert (
        serialization.truss_msgpack_deserialize(
            serialization.truss_msgpack_serialize(payload)
        )
        == payload
    )


@pytest.mark.parametrize(
    "value",
    [
        "x",
        1,
        1.5,
        True,
        None,
        {},
        [],
        datetime.now(),
        date.today(),
        time(1, 2),
        timedelta(1),
        np.zeros(2),
    ],
)
def test_is_truss_serializable_accepts_supported_types(value):
    assert serialization.is_truss_serializable(value) is True


@pytest.mark.parametrize("value", [object(), Decimal("1"), uuid.uuid4(), {1, 2}])
def test_is_truss_serializable_rejects_other_types(value):
    assert serialization.is_truss_serializable(value) is False


@pytest.mark.parametrize(
    "value, expected",
    [(np.int64(3), 3), (np.float32(1.5), 1.5), (np.arange(3), [0, 1, 2])],
)
def test_deep_numpy_encoder(value, expected):
    assert json.loads(json.dumps(value, cls=serialization.DeepNumpyEncoder)) == expected


def test_deep_numpy_encoder_rejects_unsupported_type():
    with pytest.raises(TypeError):
        json.dumps(object(), cls=serialization.DeepNumpyEncoder)
