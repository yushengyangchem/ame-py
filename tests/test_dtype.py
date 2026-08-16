import pytest

from ame_py.dtype import DType


@pytest.mark.parametrize(
    ("dtype", "bits", "bytes"),
    [
        (DType.INT4, 4, 0),
        (DType.UINT8, 8, 1),
        (DType.INT16, 16, 2),
        (DType.INT32, 32, 4),
    ],
)
def test_size(dtype, bits, bytes):
    assert DType.size_bits(dtype) == bits
    assert DType.size_bytes(dtype) == bytes


@pytest.mark.parametrize(
    ("dtype", "signed"),
    [
        (DType.INT4, True),
        (DType.INT8, True),
        (DType.INT32, True),
        (DType.UINT4, False),
        (DType.UINT8, False),
        (DType.UINT32, False),
    ],
)
def test_is_signed(dtype, signed):
    assert DType.is_signed(dtype) is signed


@pytest.mark.parametrize(
    ("dtype", "saturated"),
    [
        (DType.INT4_SAT, True),
        (DType.INT8_SAT, True),
        (DType.UINT4_SAT, True),
        (DType.UINT8_SAT, True),
        (DType.INT8, False),
        (DType.INT32, False),
    ],
)
def test_is_saturated(dtype, saturated):
    assert DType.is_saturated(dtype) is saturated
