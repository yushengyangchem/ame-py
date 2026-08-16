import numpy as np
import pytest

from ame_py import AMESimulator, DType


@pytest.fixture
def sim():
    sim = AMESimulator()
    sim.verbose = False
    return sim


def configure_gemm(sim):
    sim.msettyp(0, DType.INT8)
    sim.msettyp(1, DType.INT8)
    # M4 receives the int32 result; 4 is aligned to the int32 group_size
    sim.msettyp(4, DType.INT32)
    sim.asettyp(0, DType.INT32)


class TestSetType:
    def test_msettyp_supported(self, sim):
        assert sim.msettyp(0, DType.INT8) is True
        assert sim.Md[0] == DType.INT8
        assert sim.get_amestatus() & 1 == 0

    def test_msettyp_zeroes_register(self, sim):
        sim.M[0] = np.ones(sim.AME_NELEM)
        sim.msettyp(0, DType.INT16)
        assert np.all(sim.M[0] == 0)

    def test_msettyp_unsupported_sets_un_flag(self, sim):
        assert sim.msettyp(0, 0xDEADBEEF) is False
        assert sim.get_amestatus() & 1 == 1
        sim.clear_amestatus()
        assert sim.get_amestatus() == 0

    def test_msettyp_invalid_register(self, sim):
        with pytest.raises(ValueError):
            sim.msettyp(16, DType.INT8)

    def test_asettyp_supported(self, sim):
        assert sim.asettyp(0, DType.INT32) is True
        assert sim.Ad[0] == DType.INT32

    def test_asettyp_rejects_storage_only(self, sim):
        assert sim.asettyp(0, DType.INT4) is False
        assert sim.get_amestatus() & 1 == 1

    def test_asettyp_invalid_register(self, sim):
        with pytest.raises(ValueError):
            sim.asettyp(1, DType.INT32)


class TestLoadStore:
    def test_mls_round_trip(self, sim):
        configure_gemm(sim)
        data = np.arange(1, sim.AME_NELEM + 1, dtype=np.int64)
        sim.memory[0x1000] = data

        assert sim.mls(0, 0x1000) is True
        np.testing.assert_array_equal(sim.M[0], data)

        assert sim.mss(0, 0x2000) is True
        np.testing.assert_array_equal(sim.memory[0x2000], data)

    def test_mls_without_type_sets_un_flag(self, sim):
        assert sim.mls(0, 0x1000) is False
        assert sim.get_amestatus() & 1 == 1

    def test_mls_generates_data_for_unknown_address(self, sim):
        sim.msettyp(0, DType.INT8)
        assert sim.mls(0, 0x1000) is True
        expected = np.arange(1, sim.AME_NELEM + 1) % 128
        np.testing.assert_array_equal(sim.M[0], expected)

    def test_mls_misaligned_register(self, sim):
        sim.msettyp(3, DType.INT32)  # group_size = 4
        with pytest.raises(ValueError, match="aligned"):
            sim.mls(3, 0x1000)

    def test_mls_rm_extracts_tile(self, sim):
        sim.msettyp(0, DType.INT8)
        matrix = np.arange(1, 65, dtype=np.int64).reshape(8, 8)
        sim.memory[0x1000] = matrix

        assert sim.mls_rm(0, 0x1000) is True
        np.testing.assert_array_equal(sim.M[0].reshape(8, 8), matrix)

    def test_mss_without_type_sets_un_flag(self, sim):
        assert sim.mss(0, 0x1000) is False
        assert sim.get_amestatus() & 1 == 1

    def test_mss_out_of_range(self, sim):
        with pytest.raises(ValueError):
            sim.mss(16, 0x1000)


class TestMmulAcc:
    def test_gemm_matches_numpy(self, sim):
        configure_gemm(sim)

        rng = np.random.default_rng(42)
        A = rng.integers(0, 20, size=(8, 8))
        B = rng.integers(0, 20, size=(8, 8))
        sim.memory[0x1000] = A.flatten()
        sim.memory[0x2000] = B.flatten()

        sim.mls(0, 0x1000)
        sim.mls(1, 0x2000)
        sim.mzero_2d(0)
        assert sim.mmulacc_2d(0, 0, 1) is True

        np.testing.assert_array_equal(sim.Acc[0].reshape(8, 8), A @ B)

    def test_accumulates_across_iterations(self, sim):
        configure_gemm(sim)

        sim.memory[0x1000] = np.ones(64, dtype=np.int64)
        sim.memory[0x2000] = np.ones(64, dtype=np.int64)

        sim.mls(0, 0x1000)
        sim.mls(1, 0x2000)
        sim.mzero_2d(0)

        sim.mmulacc_2d(0, 0, 1)
        sim.mmulacc_2d(0, 0, 1)
        sim.mmulacc_2d(0, 0, 1)

        # Every element of ones(8x8) @ ones(8x8) is 8
        assert np.all(sim.Acc[0] == 3 * 8)

    def test_untyped_operands_set_un_flag(self, sim):
        sim.asettyp(0, DType.INT32)
        assert sim.mmulacc_2d(0, 0, 1) is False
        assert sim.get_amestatus() & 1 == 1

    def test_storage_only_operand_rejected(self, sim):
        sim.msettyp(0, DType.INT4)  # storage-only
        sim.msettyp(1, DType.INT8)
        sim.asettyp(0, DType.INT32)
        assert sim.mmulacc_2d(0, 0, 1) is False
        assert sim.get_amestatus() & 1 == 1

    def test_dtype_mismatch_rejected(self, sim):
        # Both aligned group starts for their own dtype, but int16 × int8
        # is mixed-precision and must be rejected.
        sim.msettyp(0, DType.INT16)
        sim.msettyp(1, DType.INT8)
        sim.asettyp(0, DType.INT32)
        assert sim.mmulacc_2d(0, 0, 1) is False
        assert sim.get_amestatus() & 1 == 1

    def test_misaligned_operand_raises(self, sim):
        sim.msettyp(3, DType.INT32)  # group_size = 4
        sim.msettyp(0, DType.INT32)
        sim.asettyp(0, DType.INT32)
        with pytest.raises(ValueError, match="misaligned"):
            sim.mmulacc_2d(0, 3, 0)

    def test_saturation_clamps_result(self, sim):
        sim.msettyp(0, DType.INT8)
        sim.msettyp(1, DType.INT8)
        sim.asettyp(0, DType.INT8_SAT)  # signed 8-bit saturation

        big = np.full(64, 120, dtype=np.int64)
        sim.memory[0x1000] = big
        sim.memory[0x2000] = big
        sim.mls(0, 0x1000)
        sim.mls(1, 0x2000)
        sim.mzero_2d(0)

        sim.mmulacc_2d(0, 0, 1)
        # Unsaturated result would be 120*120*8 = 115200, clamped to 127
        assert np.all(sim.Acc[0] == 127)


class TestMovAndZero:
    def test_mmov_a_m(self, sim):
        configure_gemm(sim)

        sim.memory[0x1000] = np.arange(1, 65, dtype=np.int64)
        sim.memory[0x2000] = np.ones(64, dtype=np.int64)
        sim.mls(0, 0x1000)
        sim.mls(1, 0x2000)
        sim.mzero_2d(0)
        sim.mmulacc_2d(0, 0, 1)
        sim.mmov_a_m(4, 0)

        np.testing.assert_array_equal(sim.M[4], sim.Acc[0])

    def test_mmov_a_m_dtype_mismatch(self, sim):
        sim.msettyp(0, DType.INT8)
        sim.asettyp(0, DType.INT32)
        assert sim.mmov_a_m(0, 0) is False
        assert sim.get_amestatus() & 1 == 1

    def test_mzero_2d(self, sim):
        sim.asettyp(0, DType.INT32)
        sim.Acc[0] = np.ones(sim.AME_NELEM)
        assert sim.mzero_2d(0) is True
        assert np.all(sim.Acc[0] == 0)

    def test_mzero_2d_untyped(self, sim):
        assert sim.mzero_2d(0) is False
        assert sim.get_amestatus() & 1 == 1


class TestState:
    def test_initial_state(self, sim):
        assert sim.amenlen == 8
        assert sim.ameudsz == 8
        assert sim.get_amestatus() == 0
        assert all(np.all(m == 0) for m in sim.M)
        assert all(d == 0 for d in sim.Md)

    def test_print_state(self, sim, capsys):
        configure_gemm(sim)
        sim.print_state()
        out = capsys.readouterr().out
        assert "AME STATE" in out
        assert "ameudsz (U) = 8 bits" in out
        assert "amenlen (N) = 8" in out
