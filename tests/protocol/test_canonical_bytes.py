"""Tests for praxis.protocol.hashing.canonical_bytes."""

import base64
import json

import numpy as np

from praxis.protocol.hashing import canonical_bytes


def test_dict_key_order_is_irrelevant() -> None:
    a = canonical_bytes({"z": 1, "a": 2, "m": 3})
    b = canonical_bytes({"a": 2, "m": 3, "z": 1})
    assert a == b


def test_list_order_is_preserved() -> None:
    a = canonical_bytes([1, 2, 3])
    b = canonical_bytes([3, 2, 1])
    assert a != b


def test_nested_dict_key_order() -> None:
    a = canonical_bytes({"x": {"z": 9, "a": 1}, "y": 0})
    b = canonical_bytes({"y": 0, "x": {"a": 1, "z": 9}})
    assert a == b


def test_numpy_array_round_trip() -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    raw = canonical_bytes(arr)

    decoded = json.loads(raw.decode("utf-8"))
    assert decoded["__np__"] is True
    assert decoded["dtype"] == "float32"
    assert decoded["shape"] == [2, 2]

    data = base64.b64decode(decoded["data_b64"])
    recovered = np.frombuffer(data, dtype=np.dtype(decoded["dtype"])).reshape(decoded["shape"])
    np.testing.assert_array_equal(recovered, arr)


def test_numpy_int_array() -> None:
    arr = np.arange(6, dtype=np.int64).reshape(2, 3)
    raw = canonical_bytes(arr)
    decoded = json.loads(raw.decode("utf-8"))
    recovered = np.frombuffer(
        base64.b64decode(decoded["data_b64"]), dtype=np.dtype(decoded["dtype"])
    ).reshape(decoded["shape"])
    np.testing.assert_array_equal(recovered, arr)


def test_float_precision_preserved() -> None:
    # repr() keeps full float precision; json.dumps alone would lose digits.
    v = 1.0000000000000002
    raw = canonical_bytes({"val": v})
    decoded = json.loads(raw.decode("utf-8"))
    # Floats are stored as repr strings under the key "val".
    assert decoded["val"] == repr(v)


def test_bool_not_treated_as_int() -> None:
    # bool must come before int in _normalize so True/False stay booleans.
    raw_true = canonical_bytes(True)
    raw_one = canonical_bytes(1)
    assert raw_true != raw_one


def test_none_encodes() -> None:
    assert canonical_bytes(None) == b"null"


def test_empty_dict_and_list() -> None:
    assert canonical_bytes({}) == b"{}"
    assert canonical_bytes([]) == b"[]"
