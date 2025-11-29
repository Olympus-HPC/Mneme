import pytest
from mneme.mneme_types import dim3


def test_dim3_default_constructor():
    d = dim3()
    assert d.x == 1
    assert d.y == 1
    assert d.z == 1


def test_dim3_custom_values():
    d = dim3(3, 4, 5)
    assert d.x == 3
    assert d.y == 4
    assert d.z == 5


def test_dim3_repr():
    d = dim3(7, 8, 9)
    assert repr(d) == "dim3(7, 8, 9)"


def test_dim3_to_dict():
    d = dim3(10, 20, 30)
    assert d.to_dict() == {"x": 10, "y": 20, "z": 30}


def test_dim3_accepts_ints_and_coerces_unsigned():
    d = dim3(-1, 0, 2**32 - 1)

    # ctypes.c_uint wraps modulo 2^32
    assert d.x == (2**32 - 1)   # -1 becomes max uint32
    assert d.y == 0
    assert d.z == (2**32 - 1)


def test_dim3_is_ctypes_structure():
    d = dim3(1, 2, 3)

    # __class__ is dim3 but inherits from ctypes.Structure
    from ctypes import Structure
    assert isinstance(d, Structure)

    # internal buffer exists (ctypes behavior)
    assert hasattr(d, "_fields_")
    assert ("x",) in [f[:1] for f in d._fields_]
