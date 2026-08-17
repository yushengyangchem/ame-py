import numpy as np
import pytest

from ame_py.dtype import DType
from ame_py.tile_unit import ELEM_BYTES, TileUnit

N = 8


@pytest.fixture
def tu():
    return TileUnit(n=N)


def _fold_ref(acc, a16, b16):
    """mmulacc.2d reference: seeded fold, ascending q, one fp32 rounding per step."""
    acc = np.float32(acc)
    for q in range(a16.shape[0]):
        prod = np.outer(a16[:, q].astype(np.float32), b16[q, :].astype(np.float32))
        acc = (acc + prod).astype(np.float32)
    return acc


def _gemm_struct_ref(tu, A, B, C=None):
    """Same numeric path as TileUnit.gemm, written independently."""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    M, K = A.shape
    P = B.shape[1]
    n = tu.N
    m8, k8, p8 = (M // n) * n, (K // n) * n, (P // n) * n
    ref = np.zeros((M, P)) if C is None else C.astype(np.float64)
    for i in range(0, m8, n):
        for j in range(0, p8, n):
            acc = np.zeros((n, n), dtype=np.float32)
            if C is not None:
                acc = C[i : i + n, j : j + n].astype(np.float32)
            for p in range(0, k8, n):
                a16 = A[i : i + n, p : p + n].astype(np.float16)
                b16 = B[p : p + n, j : j + n].astype(np.float16)
                acc = _fold_ref(acc, a16, b16)
            for p in range(k8, K):
                acc = (acc + A[i : i + n, p, None] * B[None, p, j : j + n]).astype(
                    np.float32
                )
            ref[i : i + n, j : j + n] = acc
    for i in range(M):
        for j in range(P):
            if i < m8 and j < p8:
                continue
            dot = sum(np.float32(A[i, p] * B[p, j]) for p in range(K))
            ref[i, j] = np.float32(ref[i, j] + dot)
    return ref.astype(np.float32)


class TestSpecEncodings:
    def test_dtype_encodings_match_spec_table(self):
        # FP16: exp_w=5, RNE, Inf, denormals, FP bit, 16 bits
        assert DType.FP16 == (5 << 26) | (1 << 21) | (1 << 20) | (1 << 8) | 16
        # FP32: exp_w=8, RNE, Inf, denormals, FP bit, 32 bits
        assert DType.FP32 == (8 << 26) | (1 << 21) | (1 << 20) | (1 << 8) | 32


class TestMLSST:
    def test_strided_load_reads_row_segments(self, tu):
        d, n = 24, 21
        W = np.arange(d * n, dtype=np.float32).reshape(d, n)
        tile = tu.mls_st(W.reshape(-1), 2 * n + 5, n * ELEM_BYTES)
        np.testing.assert_array_equal(tile, W[2 : 2 + N, 5 : 5 + N])

    def test_zero_stride_replicates_one_segment(self, tu):
        buf = np.arange(3 * N, dtype=np.float32)
        tile = tu.mls_st(buf, N, 0)
        np.testing.assert_array_equal(tile, np.tile(buf[N : 2 * N], (N, 1)))

    def test_non_element_stride_rejected(self, tu):
        with pytest.raises(ValueError, match="stride"):
            tu.mls_st(np.zeros(64, np.float32), 0, 6)

    def test_crossing_boundary_raises(self, tu):
        with pytest.raises(IndexError, match="boundary"):
            tu.mls_st(np.zeros(4 * N - 1, np.float32), 3 * N, N * ELEM_BYTES)


class TestMLSTST:
    def test_transposed_strided_load(self, tu):
        d, n = 16, 24
        M = np.arange(d * n, dtype=np.float32).reshape(d, n)
        tile = tu.mls_tst(M.reshape(-1), 2 * n + 4, n * ELEM_BYTES)
        # element (i, j) = element i of column-segment j
        np.testing.assert_array_equal(tile, M[2 : 2 + N, 4 : 4 + N].T)

    def test_zero_stride_column_broadcasts(self, tu):
        x = np.arange(2 * N, dtype=np.float32)
        tile = tu.mls_tst(x, N, 0)
        assert tile.shape == (N, N)
        np.testing.assert_array_equal(tile, np.tile(x[N : 2 * N, None], (1, N)))


class TestMSSST:
    def test_strided_store_round_trip(self, tu):
        rng = np.random.default_rng(2)
        n = 24
        buf = np.zeros(16 * n, dtype=np.float32)
        tile = rng.standard_normal((N, N)).astype(np.float32)
        base, stride = 2 * n + 4, n * ELEM_BYTES
        tu.mss_st(buf, base, stride, tile)
        for i in range(N):
            np.testing.assert_array_equal(buf[base + i * n : base + i * n + N], tile[i])

    def test_overlapping_segments_later_wins(self, tu):
        # stride 0: segments staged ascending, last row wins (ame-common-memory)
        buf = np.zeros(N, dtype=np.float32)
        tile = np.arange(N * N, dtype=np.float32).reshape(N, N)
        tu.mss_st(buf, 0, 0, tile)
        np.testing.assert_array_equal(buf, tile[N - 1])


class TestMconvEw:
    def test_matches_numpy_f16_rne_cast(self, tu):
        rng = np.random.default_rng(4)
        tile = rng.standard_normal((N, N)).astype(np.float32)
        np.testing.assert_array_equal(tu.mconv_ew(tile), tile.astype(np.float16))

    def test_rne_tie_cases(self, tu):
        # ULP at 1.0 is 2^-10; exact ties round to even mantissa
        below = np.float32(1.0 + 2.0**-11)
        above = np.float32(1.0 + 3 * 2.0**-12)
        tie_up = np.float32(1.0 + 3 * 2.0**-11)
        assert tu.mconv_ew(np.array([[below]], np.float32))[0, 0] == np.float16(1.0)
        assert tu.mconv_ew(np.array([[above]], np.float32))[0, 0] == np.float16(
            1.0 + 2.0**-10
        )
        assert tu.mconv_ew(np.array([[tie_up]], np.float32))[0, 0] == np.float16(
            1.0 + 2.0**-9
        )

    def test_overflow_to_inf(self, tu):
        big = np.array([[65505.0 + 512.0]], dtype=np.float32)  # > max fp16 after RNE
        with np.errstate(over="ignore"):  # the overflow is the point of the test
            out = tu.mconv_ew(big)
        assert np.isinf(out[0, 0])


class TestMmulAcc2D:
    def test_seeded_fold_matches_reference(self, tu):
        rng = np.random.default_rng(5)
        seed = rng.standard_normal((N, N)).astype(np.float32)
        a16 = rng.standard_normal((N, N)).astype(np.float16)
        b16 = rng.standard_normal((N, N)).astype(np.float16)
        np.testing.assert_array_equal(
            tu.mmulacc_2d(seed, a16, b16), _fold_ref(seed, a16, b16)
        )

    def test_zero_seed_equals_plain_fold(self, tu):
        rng = np.random.default_rng(6)
        a16 = rng.standard_normal((N, N)).astype(np.float16)
        b16 = rng.standard_normal((N, N)).astype(np.float16)
        np.testing.assert_array_equal(
            tu.mmulacc_2d(tu.mzero_2d_acc(), a16, b16),
            _fold_ref(np.zeros((N, N)), a16, b16),
        )

    def test_exact_for_small_integers(self, tu):
        rng = np.random.default_rng(7)
        a = rng.integers(0, 16, (N, N)).astype(np.float16)
        b = rng.integers(0, 16, (N, N)).astype(np.float16)
        np.testing.assert_array_equal(
            tu.mmulacc_2d(tu.mzero_2d_acc(), a, b),
            a.astype(np.int64) @ b.astype(np.int64),
        )

    def test_subnormal_products_survive_in_fp32(self, tu):
        a = np.full((N, N), np.float16(2.0**-24), dtype=np.float16)
        b = np.full((N, N), 1.0, dtype=np.float16)
        acc = tu.mmulacc_2d(tu.mzero_2d_acc(), a, b)
        assert acc[0, 0] == np.float32(N * 2.0**-24)


class TestGemm:
    SHAPES = [
        (8, 8, 8),
        (16, 16, 16),
        (24, 16, 32),
        (16, 24, 40),
        (13, 21, 9),
        (64, 48, 40),
        (33, 17, 65),
        (20, 16, 3),
        (5, 7, 6),
    ]

    def test_matches_struct_reference(self, tu):
        rng = np.random.default_rng(42)
        for M, P, K in self.SHAPES:
            A = rng.standard_normal((M, K)).astype(np.float32)
            B = rng.standard_normal((K, P)).astype(np.float32)
            C = tu.gemm(A, B)
            assert C.shape == (M, P)
            np.testing.assert_allclose(
                C, _gemm_struct_ref(tu, A, B), rtol=1e-6, atol=1e-6
            )

    def test_accumulate_into_existing_c(self, tu):
        rng = np.random.default_rng(11)
        M, P, K = 16, 24, 16
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, P)).astype(np.float32)
        C0 = rng.standard_normal((M, P)).astype(np.float32)
        got = tu.gemm(A, B, C0.copy())
        np.testing.assert_allclose(
            got, _gemm_struct_ref(tu, A, B, C0), rtol=1e-6, atol=1e-6
        )
        assert got.shape == C0.shape

    def test_accuracy_vs_fp64(self, tu):
        rng = np.random.default_rng(1)
        for M, P, K in self.SHAPES:
            A = rng.standard_normal((M, K)).astype(np.float32)
            B = rng.standard_normal((K, P)).astype(np.float32)
            C = tu.gemm(A, B)
            ref = A.astype(np.float64) @ B.astype(np.float64)
            tol = 2 * 2.0**-11 * np.sqrt(K) * (1 + np.abs(ref).max())
            assert np.max(np.abs(C - ref)) < tol

    def test_exact_for_small_integers(self, tu):
        rng = np.random.default_rng(3)
        for M, P, K in [(8, 8, 8), (16, 24, 16), (13, 21, 9)]:
            A = rng.integers(0, 16, size=(M, K)).astype(np.float32)
            B = rng.integers(0, 16, size=(K, P)).astype(np.float32)
            np.testing.assert_array_equal(tu.gemm(A, B), A @ B)

    def test_rejects_k_mismatch(self, tu):
        A = np.zeros((8, 16), dtype=np.float32)
        B = np.zeros((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="K mismatch"):
            tu.gemm(A, B)


class TestGemv:
    SHAPES = [(16, 16), (8, 8), (24, 40), (20, 13), (33, 41), (5, 7), (3, 20)]

    def test_matches_fp64_within_fp16_tolerance(self, tu):
        rng = np.random.default_rng(42)
        for d, n in self.SHAPES:
            W = rng.standard_normal((d, n)).astype(np.float32)
            x = rng.standard_normal(n).astype(np.float32)
            out = np.zeros(d, dtype=np.float32)
            tu.gemv(out, x, W.reshape(-1), n, d)
            ref = W.astype(np.float64) @ x.astype(np.float64)
            tol = 2 * 2.0**-11 * np.sqrt(n) * (1 + np.abs(ref).max())
            assert np.max(np.abs(out - ref)) < tol

    def test_exact_for_small_integers(self, tu):
        rng = np.random.default_rng(3)
        d, n = 24, 16
        W = rng.integers(0, 16, size=(d, n)).astype(np.float32)
        x = rng.integers(0, 16, size=n).astype(np.float32)
        out = np.zeros(d, dtype=np.float32)
        tu.gemv(out, x, W.reshape(-1), n, d)
        np.testing.assert_array_equal(out, W @ x)

    def test_rejects_mismatched_buffers(self, tu):
        x = np.zeros(4, dtype=np.float32)
        out = np.zeros(8, dtype=np.float32)
        w = np.zeros(16, dtype=np.float32)
        with pytest.raises(ValueError, match="match"):
            tu.gemv(out, x, w, 4, 8)
