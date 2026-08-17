"""AME data type encodings (32-bit specifiers)."""


class DType:
    """AME data type encodings."""

    # Standard integer types
    INT4 = 0x40000004
    UINT4 = 0x00000004
    INT8 = 0x40000008
    UINT8 = 0x00000008
    INT16 = 0x40000010
    UINT16 = 0x00000010
    INT32 = 0x40000020
    UINT32 = 0x00000020

    # Saturating integer types
    INT4_SAT = 0x60000004
    UINT4_SAT = 0x20000004
    INT8_SAT = 0x60000008
    UINT8_SAT = 0x20000008

    # IEEE floating-point class (spec datatype table): bit 8 = FP,
    # bits 30:26 = exponent width, bit 21 = Inf, bit 20 = denormals,
    # bits 7:0 = size in bits, rounding = RNE (bits 25:22 = 0).
    FP16 = 0x14300110
    FP32 = 0x20300120

    @staticmethod
    def size_bits(dtype: int) -> int:
        """Extract the width in bits of a data type (bits[7:0])."""
        return dtype & 0xFF

    @staticmethod
    def is_signed(dtype: int) -> bool:
        """Check whether an integer data type is signed (bit 30)."""
        return (dtype >> 30) & 1 == 1

    @staticmethod
    def is_saturated(dtype: int) -> bool:
        """Check whether a data type uses saturating arithmetic (bit 29)."""
        return (dtype >> 29) & 1 == 1
