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
        datetime(2024, 1, 2, 3, 4, 5, 678901),
        datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        date(2024, 1, 2),
        time(3, 4, 5),
        timedelta(days=1, seconds=2, microseconds=3),
        Decimal("1.25"),
        uuid.uuid4(),
    ],
)
def test_msgpack_roundtrip_preserves_rich_types(value):
    data = serialization.truss_msgpack_serialize({"value": value})

    assert serialization.truss_msgpack_deserialize(data) == {"value": value}


def test_msgpack_roundtrip_preserves_numpy_arrays():
    array = np.arange(6, dtype=np.float32).reshape(2, 3)

    data = serialization.truss_msgpack_serialize({"array": array})
    deserialized = serialization.truss_msgpack_deserialize(data)

    assert np.array_equal(deserialized["array"], array)
    assert deserialized["array"].dtype == array.dtype


def test_encoder_renders_utc_datetime_with_z_suffix():
    encoded = serialization._truss_msgpack_encoder(
        datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    )

    assert encoded == {b"__dt_datetime_iso__": True, b"data": "2024-01-02T03:04:05Z"}


def test_encoder_rejects_timezone_aware_time():
    with pytest.raises(ValueError, match="Cannot represent timezone-aware times."):
        serialization._truss_msgpack_encoder(time(3, 4, 5, tzinfo=timezone.utc))


def test_encoder_delegates_unknown_types_to_chain():
    sentinel = object()

    assert serialization._truss_msgpack_encoder(sentinel, chain=lambda o: o) is sentinel


def test_encoder_returns_unknown_types_without_chain():
    sentinel = {"plain": "dict"}

    assert serialization._truss_msgpack_encoder(sentinel) is sentinel


def test_decoder_passes_through_unknown_payloads():
    payload = {b"other": 1}

    assert serialization._truss_msgpack_decoder(payload) is payload
    assert serialization._truss_msgpack_decoder(payload, chain=lambda o: o) is payload


def test_decoder_passes_through_incomplete_payloads():
    payload = {b"__decimal__": True}

    assert serialization._truss_msgpack_decoder(payload) is payload
    assert serialization._truss_msgpack_decoder(payload, chain=lambda o: o) is payload


@pytest.mark.parametrize(
    "value, expected",
    [
        ("string", True),
        (1, True),
        (1.5, True),
        (True, True),
        (None, True),
        ({"key": "value"}, True),
        ([1, 2], True),
        (datetime(2024, 1, 2), True),
        (date(2024, 1, 2), True),
        (time(3, 4), True),
        (timedelta(seconds=1), True),
        (np.array([1, 2]), True),
        (Decimal("1.5"), False),
        (object(), False),
    ],
)
def test_is_truss_serializable(value, expected):
    assert serialization.is_truss_serializable(value) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.int64(3), 3),
        (np.float32(1.5), 1.5),
        (np.array([[1, 2], [3, 4]]), [[1, 2], [3, 4]]),
    ],
)
def test_deep_numpy_encoder_converts_numpy_types(value, expected):
    assert json.loads(json.dumps(value, cls=serialization.DeepNumpyEncoder)) == expected


def test_deep_numpy_encoder_rejects_unsupported_types():
    with pytest.raises(TypeError):
        json.dumps(object(), cls=serialization.DeepNumpyEncoder)
