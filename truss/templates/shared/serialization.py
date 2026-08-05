import json
import uuid
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

if TYPE_CHECKING:
    from numpy.typing import NDArray


JSONType = Union[str, int, float, bool, None, list["JSONType"], dict[str, "JSONType"]]
MsgPackType = Union[
    str,
    int,
    float,
    bool,
    None,
    date,
    Decimal,
    datetime,
    time,
    timedelta,
    uuid.UUID,
    "NDArray",
    list["MsgPackType"],
    dict[str, "MsgPackType"],
]


# mostly cribbed from django.core.serializer.DjangoJSONEncoder
def _truss_msgpack_encoder(
    obj: Union[Decimal, date, time, timedelta, uuid.UUID, dict],
    chain: Optional[Callable] = None,
) -> dict:
    if isinstance(obj, datetime):
        r = obj.isoformat()
        if r.endswith("+00:00"):
            r = r[:-6] + "Z"
        return {b"__dt_datetime_iso__": True, b"data": r}
    elif isinstance(obj, date):
        r = obj.isoformat()
        return {b"__dt_date_iso__": True, b"data": r}
    elif isinstance(obj, time):
        if obj.utcoffset() is not None:
            raise ValueError("Cannot represent timezone-aware times.")
        r = obj.isoformat()
        return {b"__dt_time_iso__": True, b"data": r}
    elif isinstance(obj, timedelta):
        return {
            b"__dt_timedelta__": True,
            b"data": (obj.days, obj.seconds, obj.microseconds),
        }
    elif isinstance(obj, Decimal):
        return {b"__decimal__": True, b"data": str(obj)}
    elif isinstance(obj, uuid.UUID):
        return {b"__uuid__": True, b"data": str(obj)}
    else:
        return obj if chain is None else chain(obj)


def _unpack_dtype(dtype: Any) -> Any:
    import numpy as np

    if isinstance(dtype, (list, tuple)):
        dtype = [
            (subdtype[0], _unpack_dtype(subdtype[1])) + tuple(subdtype[2:])
            for subdtype in dtype
        ]
    return np.dtype(dtype)


def _tostr(x: Any) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def _numpy_decoder(obj: Any) -> Any:
    """Pickle-free replacement for `msgpack_numpy.decode`.

    `msgpack_numpy.decode` calls `pickle.loads` for object-dtype (`kind == b"O"`)
    payloads, which is arbitrary code execution when the bytes are untrusted.
    Object dtypes are rejected here instead.
    """
    import numpy as np

    try:
        if b"nd" in obj:
            if obj[b"nd"] is True:
                kind = obj.get(b"kind")
                if kind == b"O":
                    raise ValueError(
                        "Deserializing object-dtype numpy arrays is not supported, "
                        "because it requires unpickling untrusted data."
                    )
                if kind == b"V":
                    descr = [
                        tuple(_tostr(t) if isinstance(t, bytes) else t for t in d)
                        for d in obj[b"type"]
                    ]
                else:
                    descr = obj[b"type"]
                dtype = _unpack_dtype(descr)
            else:
                dtype = _unpack_dtype(obj[b"type"])

            if dtype.hasobject:
                raise ValueError(
                    "Deserializing object-dtype numpy arrays is not supported, "
                    "because it requires unpickling untrusted data."
                )
            if obj[b"nd"] is True:
                return np.ndarray(
                    buffer=obj[b"data"], dtype=dtype, shape=tuple(obj[b"shape"])
                )
            return np.frombuffer(obj[b"data"], dtype=dtype)[0]
        elif b"complex" in obj:
            return complex(_tostr(obj[b"data"]))
        else:
            return obj
    except KeyError:
        return obj


def _truss_msgpack_decoder(obj: Any, chain=None):
    try:
        if b"__dt_datetime_iso__" in obj:
            return datetime.fromisoformat(obj[b"data"])
        elif b"__dt_date_iso__" in obj:
            return date.fromisoformat(obj[b"data"])
        elif b"__dt_time_iso__" in obj:
            return time.fromisoformat(obj[b"data"])
        elif b"__dt_timedelta__" in obj:
            days, seconds, microseconds = obj[b"data"]
            return timedelta(days=days, seconds=seconds, microseconds=microseconds)
        elif b"__decimal__" in obj:
            return Decimal(obj[b"data"])
        elif b"__uuid__" in obj:
            return uuid.UUID(obj[b"data"])
        else:
            return obj if chain is None else chain(obj)
    except KeyError:
        return obj if chain is None else chain(obj)


# this json object is JSONType + np.array + datetime
def is_truss_serializable(obj: Any) -> bool:
    import numpy as np

    # basic JSON types
    if isinstance(obj, (str, int, float, bool, type(None), dict, list)):
        return True
    elif isinstance(obj, (datetime, date, time, timedelta)):
        return True
    elif isinstance(obj, np.ndarray):
        return True
    else:
        return False


def _numpy_encoder(obj: Any) -> Any:
    import msgpack_numpy as mp_np
    import numpy as np

    if isinstance(obj, np.ndarray) and obj.dtype.hasobject:
        raise ValueError(
            "Serializing object-dtype numpy arrays is not supported, because they "
            "can only be represented as pickled data."
        )
    return mp_np.encode(obj)


def truss_msgpack_serialize(obj: MsgPackType) -> bytes:
    import msgpack

    return msgpack.packb(
        obj, default=lambda x: _truss_msgpack_encoder(x, chain=_numpy_encoder)
    )


def truss_msgpack_deserialize(data: bytes) -> MsgPackType:
    import msgpack

    return msgpack.unpackb(
        data,
        object_hook=lambda x: _truss_msgpack_decoder(x, chain=_numpy_decoder),
        strict_map_key=True,
    )


class DeepNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)
