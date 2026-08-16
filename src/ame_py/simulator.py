"""AMESimulator: a NumPy-based simulator of the Attached Matrix Extension."""

import math

import numpy as np

from ame_py.dtype import DType


class AMESimulator:
    """
    AME simulator for understanding and verifying AME instruction behavior.

    Hardware parameters (mirroring the spec defaults):
    - AME_NELEM = 64 (N = 8)
    - AME_UNIT_DATATYPE_SIZE = 8
    - AME_NUM_M_REGS = 16
    - AME_NUM_ACC_REGS = 1
    """

    def __init__(self):
        # Hardware parameters
        self.AME_NELEM = 64
        self.N = math.isqrt(self.AME_NELEM)
        self.AME_UNIT_DATATYPE_SIZE = 8
        self.AME_NUM_M_REGS = 16
        self.AME_NUM_ACC_REGS = 1

        # Register files: M holds matrix data, Md holds its data type.
        #
        # HACK: unpacked register representation — every element slot is an
        # int64 regardless of Md. Real hardware stores elements at their
        # declared width (packed, e.g. 2x INT4 per 8-bit unit slot). Here the
        # int64 container only mimics the hardware's wide compute path; the
        # simulated element width is tracked per register by Md and applied
        # only at compute boundaries (see mmulacc_2d). Do NOT narrow this
        # container — intermediate results would wrap and break semantics.
        self.M = [
            np.zeros(self.AME_NELEM, dtype=np.int64) for _ in range(self.AME_NUM_M_REGS)
        ]
        self.Md = [0] * self.AME_NUM_M_REGS

        # Acc holds accumulators, Ad holds its data type
        # HACK: same unpacked int64 representation as M (see note above).
        self.Acc = [
            np.zeros(self.AME_NELEM, dtype=np.int64)
            for _ in range(self.AME_NUM_ACC_REGS)
        ]
        self.Ad = [0] * self.AME_NUM_ACC_REGS

        # CSRs: amestatus bit 0 is the UN (unsupported) flag
        self.amestatus = 0
        self.amenlen = self.N
        self.ameudsz = self.AME_UNIT_DATATYPE_SIZE

        # HACK: memory is a Python dict (address -> numpy array), not a
        # byte-addressable contiguous space. No real addresses, no packing,
        # no endianness. Only the load/store *semantics* are simulated.
        self.memory: dict[int, np.ndarray] = {}

        # Debug
        self.verbose = True
        self.instruction_count = 0

        self._SUPPORTED_DTYPES = [
            DType.INT4,
            DType.INT8,
            DType.INT16,
            DType.INT32,
            DType.UINT4,
            DType.UINT8,
            DType.UINT16,
            DType.UINT32,
            DType.INT4_SAT,
            DType.INT8_SAT,
            DType.FP32,
        ]
        self._DTYPE_NAMES = {
            DType.INT8: "int8",
            DType.INT16: "int16",
            DType.INT32: "int32",
            DType.UINT8: "uint8",
            DType.FP32: "fp32",
        }

    # ========== Helpers ==========

    def _dtype_to_numpy(self, dtype: int) -> np.dtype:
        """Convert an AME data type to the closest NumPy dtype."""
        size = DType.size_bytes(dtype)
        if DType.is_signed(dtype):
            return np.dtype(f"int{size * 8}")
        return np.dtype(f"uint{size * 8}")

    def _get_group_size(self, dtype: int) -> int:
        """group_size = dtype_size_bits / U"""
        dtype_bits = DType.size_bits(dtype)
        return max(1, dtype_bits // self.AME_UNIT_DATATYPE_SIZE)

    def _get_pack_factor(self, dtype: int) -> int:
        """pack_factor = U / dtype_size_bits (storage-only types)"""
        dtype_bits = DType.size_bits(dtype)
        if dtype_bits < self.AME_UNIT_DATATYPE_SIZE:
            return self.AME_UNIT_DATATYPE_SIZE // dtype_bits
        return 1

    def _is_storage_only(self, dtype: int) -> bool:
        """Storage-only types are narrower than the unit datatype (bits < U)."""
        return DType.size_bits(dtype) < self.AME_UNIT_DATATYPE_SIZE

    def _check_register_alignment(self, reg_idx: int, dtype: int) -> bool:
        """Check that a register index is aligned to its group_size."""
        if self._is_storage_only(dtype):
            return True
        return reg_idx % self._get_group_size(dtype) == 0

    def _check_m_reg_range(self, reg_idx: int) -> bool:
        return 0 <= reg_idx < self.AME_NUM_M_REGS

    def _check_acc_reg_range(self, reg_idx: int) -> bool:
        return 0 <= reg_idx < self.AME_NUM_ACC_REGS

    def _saturate(self, value: int, dtype: int) -> int:
        """Clamp a value to the range of a saturating data type."""
        if not DType.is_saturated(dtype):
            return value

        bits = DType.size_bits(dtype)
        if DType.is_signed(dtype):
            max_val, min_val = (1 << (bits - 1)) - 1, -(1 << (bits - 1))
        else:
            max_val, min_val = (1 << bits) - 1, 0
        return max(min_val, min(max_val, value))

    # ========== Core instructions ==========

    def msettyp(self, md: int, xs1: int) -> bool:
        """
        msettyp md, xs1 - set the data type of an M register.

        Reads the 32-bit type specifier from xs1, checks that it is
        supported, writes it to Md[md] and zeroes M[md]. Unsupported
        types set amestatus.UN.
        """
        dtype = xs1 & 0xFFFFFFFF

        if not self._check_m_reg_range(md):
            raise ValueError(f"Invalid M register index: {md}")

        if dtype == 0 or dtype not in self._SUPPORTED_DTYPES:
            self.amestatus |= 1
            if self.verbose:
                print(f"  [WARN] msettyp m{md}, 0x{dtype:08X} → UNSUPPORTED")
            return False

        if self._is_storage_only(dtype) and self._get_pack_factor(dtype) < 1:
            self.amestatus |= 1
            return False

        self.Md[md] = dtype
        self.M[md] = np.zeros(self.AME_NELEM, dtype=np.int64)
        self.amestatus &= ~1

        if self.verbose:
            dtype_name = self._DTYPE_NAMES.get(dtype, f"0x{dtype:08X}")
            pack_info = (
                f" (pack_factor={self._get_pack_factor(dtype)})"
                if self._is_storage_only(dtype)
                else ""
            )
            group_info = (
                f" (group_size={self._get_group_size(dtype)})"
                if not self._is_storage_only(dtype) and self._get_group_size(dtype) > 1
                else ""
            )
            print(
                f"  [OK] msettyp m{md}, {dtype_name} 0x{dtype:08X}{pack_info}{group_info}"
            )

        return True

    def asettyp(self, ad: int, xs1: int) -> bool:
        """
        asettyp ad, xs1 - set the data type of an Acc register.

        Same as msettyp, but operates on Acc/Ad and only accepts
        compute types (storage-only types are rejected).
        """
        dtype = xs1 & 0xFFFFFFFF

        if not self._check_acc_reg_range(ad):
            raise ValueError(f"Invalid Acc register index: {ad}")

        if dtype == 0 or self._is_storage_only(dtype):
            self.amestatus |= 1
            if self.verbose:
                print(
                    f"  [WARN] asettyp acc{ad}, 0x{dtype:08X} → UNSUPPORTED (storage-only not allowed)"
                )
            return False

        self.Ad[ad] = dtype
        self.Acc[ad] = np.zeros(self.AME_NELEM, dtype=np.int64)
        self.amestatus &= ~1

        if self.verbose:
            dtype_name = self._DTYPE_NAMES.get(dtype, f"0x{dtype:08X}")
            print(f"  [OK] asettyp acc{ad}, {dtype_name} 0x{dtype:08X}")

        return True

    def mls(self, md: int, address: int) -> bool:
        """
        mls md, address - load a tile from memory into an M register.

        Reads AME_NELEM elements per register from memory in an
        implementation-defined layout, filling group_size registers.
        """
        if not self._check_m_reg_range(md):
            raise ValueError(f"Invalid M register index: {md}")

        dtype = self.Md[md]
        if dtype == 0:
            self.amestatus |= 1
            return False

        group_size = self._get_group_size(dtype)
        if md % group_size != 0:
            raise ValueError(f"M{md} not aligned to group_size={group_size}")

        # HACK: fabricates synthetic data (1..64, clipped per dtype) when the
        # address was never written. Real hardware would load whatever bytes
        # are in memory; this simulator just makes up deterministic values.
        if address not in self.memory:
            data = np.arange(1, self.AME_NELEM + 1, dtype=np.int64)
            if dtype == DType.INT8:
                data = data % 128
            elif dtype == DType.INT16:
                data = data % 32768
            elif dtype == DType.INT32:
                data = data % 1000
            self.memory[address] = data

        data = self.memory[address]
        for i in range(min(group_size, len(data) // self.AME_NELEM)):
            start = i * self.AME_NELEM
            # HACK: no narrowing on write-back — values are stored as-is in
            # the int64 container. Hardware would wrap/saturate to the Md
            # width at load time; we defer that to compute time instead.
            self.M[md + i] = data[start : start + self.AME_NELEM].copy()

        self.amestatus &= ~1
        if self.verbose:
            print(
                f"  [OK] mls m{md}, 0x{address:08X} (loaded {group_size} register(s))"
            )

        return True

    def mls_rm(self, md: int, address: int, rows: int = 8, cols: int = 8) -> bool:
        """
        mls.rm md, address - load a row-major tile from memory.

        Extracts an N×N tile from a 2D row-major matrix, starting
        at element (0, 0).
        """
        if not self._check_m_reg_range(md):
            raise ValueError(f"Invalid M register index: {md}")

        dtype = self.Md[md]
        if dtype == 0:
            self.amestatus |= 1
            return False

        # HACK: fabricates a synthetic row-major matrix if the address was
        # never written (same caveat as mls above).
        if address not in self.memory:
            data = np.arange(1, rows * cols + 1, dtype=np.int64).reshape(rows, cols)
            self.memory[address] = data

        matrix = self.memory[address]
        tile = matrix[: self.N, : self.N].flatten()
        # HACK: no narrowing on write-back (see mls above).
        self.M[md] = tile.copy()

        self.amestatus &= ~1
        if self.verbose:
            print(
                f"  [OK] mls.rm m{md}, 0x{address:08X} (row-major {self.N}x{self.N} tile)"
            )

        return True

    def mmulacc_2d(self, acc: int, ms1: int, ms2: int) -> bool:
        """
        mmulacc.2d acc, ms1, ms2 - matrix multiply-accumulate.

        Computes Acc[acc] += M[ms1] × M[ms2] over two N×N tiles and
        applies saturation when the accumulator type requires it.
        """
        if not self._check_acc_reg_range(acc):
            raise ValueError(f"Invalid Acc register index: {acc}")
        if not self._check_m_reg_range(ms1) or not self._check_m_reg_range(ms2):
            raise ValueError("Invalid M register index")

        dtype1, dtype2, dtype_acc = self.Md[ms1], self.Md[ms2], self.Ad[acc]
        if dtype1 == 0 or dtype2 == 0 or dtype_acc == 0:
            self.amestatus |= 1
            return False

        if self._is_storage_only(dtype1) or self._is_storage_only(dtype2):
            self.amestatus |= 1
            if self.verbose:
                print(
                    f"  [WARN] mmulacc.2d acc{acc}, m{ms1}, m{ms2} → storage-only dtype not allowed"
                )
            return False

        group_size1, group_size2 = (
            self._get_group_size(dtype1),
            self._get_group_size(dtype2),
        )
        if ms1 % group_size1 != 0 or ms2 % group_size2 != 0:
            raise ValueError(
                f"Register misaligned: m{ms1}%{group_size1} or m{ms2}%{group_size2}"
            )

        A = self.M[ms1].reshape(self.N, self.N)
        B = self.M[ms2].reshape(self.N, self.N)
        # HACK: computes in the full int64 container width, which models the
        # hardware's wide MAC path (e.g. INT8xINT8 -> INT32 accumulate).
        # Real hardware never lets intermediates wrap at the storage width;
        # neither does this simulator. FP types are treated as integers —
        # no IEEE-754 encoding/rounding is modeled.
        C = A @ B

        acc_data = self.Acc[acc].reshape(self.N, self.N)
        result_flat = (acc_data + C).flatten()
        # The ONLY narrowing point in the simulator: saturating accumulator
        # types are clamped on write-back, mirroring hardware behavior.
        if DType.is_saturated(dtype_acc):
            for i in range(len(result_flat)):
                result_flat[i] = self._saturate(int(result_flat[i]), dtype_acc)

        self.Acc[acc] = result_flat

        self.amestatus &= ~1
        if self.verbose:
            print(f"  [OK] mmulacc.2d acc{acc}, m{ms1}, m{ms2}")
            print(f"     A = {A.round(1)}")
            print(f"     B = {B.round(1)}")
            print(f"     A×B = {C.round(1)}")

        return True

    def mmov_a_m(self, md: int, acc: int) -> bool:
        """mmov.a.m md, acc - move data from an Acc register to an M register."""
        if not self._check_m_reg_range(md) or not self._check_acc_reg_range(acc):
            raise ValueError("Invalid register index")

        dtype_dest, dtype_src = self.Md[md], self.Ad[acc]
        if dtype_dest == 0 or dtype_src == 0:
            self.amestatus |= 1
            return False

        if DType.size_bits(dtype_dest) != DType.size_bits(dtype_src):
            self.amestatus |= 1
            if self.verbose:
                print(f"  [WARN] mmov.a.m m{md}, acc{acc} → dtype mismatch")
            return False

        group_size = self._get_group_size(dtype_dest)
        if md % group_size != 0:
            raise ValueError(f"M{md} not aligned to group_size={group_size}")

        # HACK: no narrowing on move — the int64 payload is copied verbatim
        # even if the destination Md width is narrower than the source values.
        self.M[md] = self.Acc[acc].copy()
        self.amestatus &= ~1

        if self.verbose:
            print(f"  [OK] mmov.a.m m{md}, acc{acc}")

        return True

    def mss(self, ms1: int, address: int) -> bool:
        """mss ms1, address - store a tile from an M register to memory."""
        if not self._check_m_reg_range(ms1):
            raise ValueError(f"Invalid M register index: {ms1}")

        dtype = self.Md[ms1]
        if dtype == 0:
            self.amestatus |= 1
            return False

        self.memory[address] = self.M[ms1].copy()
        self.amestatus &= ~1

        if self.verbose:
            print(f"  [OK] mss m{ms1}, 0x{address:08X}")

        return True

    def mzero_2d(self, acc: int) -> bool:
        """mzero.2d acc - zero an Acc register."""
        if not self._check_acc_reg_range(acc):
            raise ValueError(f"Invalid Acc register index: {acc}")

        if self.Ad[acc] == 0:
            self.amestatus |= 1
            return False

        self.Acc[acc] = np.zeros(self.AME_NELEM, dtype=np.int64)
        self.amestatus &= ~1

        if self.verbose:
            print(f"  [OK] mzero.2d acc{acc}")

        return True

    # ========== CSRs ==========

    def get_amestatus(self) -> int:
        """Read the amestatus CSR."""
        return self.amestatus

    def clear_amestatus(self) -> None:
        """Clear amestatus (reset the UN flag)."""
        self.amestatus = 0

    # ========== State dump ==========

    def print_state(self) -> None:
        """Print the current AME state."""
        print("\n" + "=" * 60)
        print("AME STATE")
        print("=" * 60)
        print(f"ameudsz (U) = {self.ameudsz} bits")
        print(f"amenlen (N) = {self.amenlen}")
        print(f"amestatus.UN = {self.amestatus & 1}")
        print(f"instruction count = {self.instruction_count}")

        print("\n--- M Registers ---")
        for i in range(min(8, self.AME_NUM_M_REGS)):
            dtype_name = self._DTYPE_NAMES.get(self.Md[i], "uninit")
            if self.Md[i] != 0:
                data = self.M[i].reshape(self.N, self.N)
                print(f"M{i} ({dtype_name}):\n{data.round(1)}")
            else:
                print(f"M{i}: (uninitialized)")

        print("\n--- Acc Registers ---")
        for i in range(self.AME_NUM_ACC_REGS):
            dtype_name = self._DTYPE_NAMES.get(self.Ad[i], "uninit")
            if self.Ad[i] != 0:
                data = self.Acc[i].reshape(self.N, self.N)
                print(f"acc{i} ({dtype_name}):\n{data.round(1)}")
            else:
                print(f"acc{i}: (uninitialized)")
        print("=" * 60 + "\n")
