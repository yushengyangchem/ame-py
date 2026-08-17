"""Tiled GEMM simulated strictly with the AME specification's instruction set.

Numeric model of mmulacc.2d under ame-common-matmul: the fold is seeded with
the pre-instruction accumulator element and products are added one at a time
(no separate reduction); FP16 x FP16 products are exact in FP32 (11-bit
mantissas, exponent headroom, subnormals included), so rounding happens only
at each FP32 accumulation step.  This module documents ascending-q order as
its deterministic permutation.

Memory model: `buf` arguments are flat float32 NumPy arrays, `base` is an
element index, and `stride` follows the spec in BYTES (so a row-major matrix
of width n uses stride = 4*n).  A stride that is not a whole number of
elements is rejected.  Base addresses are naturally element-aligned by
construction.  Like the spec's GEMM examples ("all examples handle only
interior squares"), boundary rows/columns/remainders that an N x N square
cannot cover are finished in scalar FP32 by software.

Register allocation used by the kernels (U = 8 bits, AME_NELEM = 64, 16 M
registers).  The span of every group is DERIVED, not chosen: it follows
group_size = dtype_bits / U once msettyp programs Md[md]; a square always
holds AME_NELEM = 64 elements, so an FP32 group spans 4 registers (2048 bits)
and an FP16 group spans 2 (1024 bits):

    m0-m3    A square, FP32 (group_size 4, from msettyp m0, FP32)
    m4-m7    B square, FP32 (group_size 4)
    m8-m9    A square, FP16 (mconv.ew m8, m0: 4-reg source, 2-reg dest)
    m10-m11  B square, FP16 (mconv.ew m10, m4)
    m12-m15  C staging, FP32 (group_size 4; mls.st/mss.st + mmov.a.m/mmov.m.a)
    acc0     FP32 accumulator (mzero.2d.acc / mmulacc.2d)
"""

import numpy as np

ELEM_BYTES = 4  # sizeof(fp32): the spec strides are in bytes


class TileGEMM:
    """An N x N tile unit programmed with spec AME instructions only."""

    def __init__(self, n: int = 8):
        """`n` is amenlen = sqrt(AME_NELEM); 8 matches AMESimulator defaults."""
        self.N = n

    # ========== Spec instructions (software model) ==========

    def mls_st(self, buf: np.ndarray, base: int, stride_bytes: int) -> np.ndarray:
        """mls.st md, (rs1), rs2 - load a square from strided memory.

        N row segments; segment i starts at base + i*stride_bytes and holds N
        elements (programming_model.adoc, `.st` segment rules).
        """
        s = self._stride_elems(stride_bytes)
        rows = []
        for i in range(self.N):
            off = base + i * s
            self._check_segment(buf, off)
            rows.append(buf[off : off + self.N])
        return np.stack(rows).astype(np.float32)

    def mls_tst(self, buf: np.ndarray, base: int, stride_bytes: int) -> np.ndarray:
        """mls.tst md, (rs1), rs2 - load a square from transposed strided memory.

        N column segments; element (i, j) is element i of segment j, so
        tile[i, j] = buf[base + j*s + i].  With stride 0 every segment reads
        the same N elements and the vector is broadcast down the columns.
        """
        s = self._stride_elems(stride_bytes)
        tile = np.empty((self.N, self.N), dtype=np.float32)
        for j in range(self.N):
            off = base + j * s
            self._check_segment(buf, off)
            tile[:, j] = buf[off : off + self.N]
        return tile

    def mss_st(
        self, buf: np.ndarray, base: int, stride_bytes: int, tile: np.ndarray
    ) -> None:
        """mss.st ms1, (rs1), rs2 - store a square to strided memory.

        Segments are staged in ascending order, so for overlapping segments
        (e.g. stride 0) bytes of a later segment win, per ame-common-memory.
        """
        s = self._stride_elems(stride_bytes)
        for i in range(self.N):
            off = base + i * s
            self._check_segment(buf, off)
            buf[off : off + self.N] = tile[i]

    def mconv_ew(self, tile: np.ndarray) -> np.ndarray:
        """mconv.ew md, ms1 with Md FP32 -> FP16; NumPy's cast is RNE."""
        return tile.astype(np.float16)

    def mzero_2d_acc(self) -> np.ndarray:
        """mzero.2d.acc acc0 - an all-zero FP32 accumulator square."""
        return np.zeros((self.N, self.N), dtype=np.float32)

    def mmulacc_2d(
        self, acc: np.ndarray, a16: np.ndarray, b16: np.ndarray
    ) -> np.ndarray:
        """mmulacc.2d acc0, ms1, ms2 - C = C + A x B (ame-common-matmul).

        The fold is seeded with the accumulator element; each FP16 product is
        exact in FP32 and added in ascending q, one rounding per step.
        """
        acc = acc.astype(np.float32)
        for q in range(self.N):
            prod = np.outer(a16[:, q].astype(np.float32), b16[q, :].astype(np.float32))
            acc = (acc + prod).astype(np.float32)
        return acc

    def mmov_m_a(self, acc: np.ndarray) -> np.ndarray:
        """mmov.m.a md, acc0 - move an accumulator square to an FP32 M group."""
        # HACK: no data is actually moved; NumPy has no separate register
        # storage, so both sides are the same ndarray and this is only a
        # value-preserving copy that marks the instruction boundary.
        # dtype-mismatch (amestatus.UN) is simulator.py's job, not modeled here.
        return acc.astype(np.float32)

    def mmov_a_m(self, m_sq: np.ndarray) -> np.ndarray:
        """mmov.a.m acc0, ms - move an FP32 M square into the accumulator."""
        # HACK: same as mmov_m_a - a copy, not a real M-register transfer.
        return m_sq.astype(np.float32)

    # ========== Kernels ==========

    def gemm(
        self, A: np.ndarray, B: np.ndarray, C: np.ndarray | None = None
    ) -> np.ndarray:
        """C[M, P] (+)= A[M, K] @ B[K, P] over N x N squares.

        Interior squares follow the spec's portable GEMM triple loop: per
        (i, j) the k loop loads A and B squares (mls.st), converts both to
        FP16 (mconv.ew) and accumulates (mmulacc.2d); the result goes back
        through mmov.m.a + mss.st.  K remainders inside interior squares and
        all boundary rows/columns are finished in scalar FP32.
        """
        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)
        M, K = A.shape
        if B.shape[0] != K:
            raise ValueError("K mismatch between A and B")
        P = B.shape[1]
        fresh = C is None
        C = (
            np.zeros((M, P), np.float32)
            if fresh
            else np.ascontiguousarray(C, dtype=np.float32)
        )
        N = self.N
        m8, k8, p8 = (M // N) * N, (K // N) * N, (P // N) * N
        Af, Bf, Cf = A.reshape(-1), B.reshape(-1), C.reshape(-1)

        for i in range(0, m8, N):
            for j in range(0, p8, N):
                # seed: mzero.2d.acc, or mls.st C-square + mmov.a.m for +=
                if fresh:
                    acc = self.mzero_2d_acc()
                else:
                    acc = self.mmov_a_m(self.mls_st(Cf, i * P + j, P * ELEM_BYTES))
                for p in range(0, k8, N):
                    a16 = self.mconv_ew(self.mls_st(Af, i * K + p, K * ELEM_BYTES))
                    b16 = self.mconv_ew(self.mls_st(Bf, p * P + j, P * ELEM_BYTES))
                    acc = self.mmulacc_2d(acc, a16, b16)  # mmulacc.2d
                # K remainder rows inside these interior squares (scalar)
                for p in range(k8, K):
                    prod = A[i : i + N, p, None] * B[None, p, j : j + N]
                    acc = (acc + prod.astype(np.float32)).astype(np.float32)
                # mmov.m.a + mss.st write the interior square back
                self.mss_st(Cf, i * P + j, P * ELEM_BYTES, self.mmov_m_a(acc))

        # boundary rows/columns a full square cannot cover: scalar FP32
        for i in range(M):
            if i < m8 and p8 == P:
                continue  # row fully covered by interior squares
            for j in range(P):
                if i < m8 and j < p8:
                    continue  # interior square element
                Cf[i * P + j] = np.float32(
                    Cf[i * P + j] + self._dot_fp32(Af, i, K, Bf, P, j)
                )
        return C

    def matmul(
        self, xout: np.ndarray, x: np.ndarray, w: np.ndarray, n: int, d: int
    ) -> None:
        """W (d, n) @ x (n,) -> xout (d,), the reference C `matmul` in spec form.

        Interior row tiles: the W square loads with mls.st (stride 4n) and x
        loads as a column-broadcast square with mls.tst stride 0, so every
        column of the accumulated square holds the same partial dot product.
        K remainders and boundary rows stay scalar FP32.
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        xout = np.ascontiguousarray(xout, dtype=np.float32)
        w = np.ascontiguousarray(w, dtype=np.float32).reshape(-1)
        if xout.shape != (d,) or x.shape != (n,) or w.size != d * n:
            raise ValueError("buffer shapes do not match (n, d)")

        N = self.N
        trow = (d // N) * N
        tcol = (n // N) * N
        xout[:] = 0

        for row in range(0, trow, N):
            acc = self.mzero_2d_acc()
            for col in range(0, tcol, N):
                w16 = self.mconv_ew(self.mls_st(w, row * n + col, n * ELEM_BYTES))
                # mls.tst with stride 0: square[q, j] = x[col + q] for all j
                x16 = self.mconv_ew(self.mls_tst(x, col, 0))
                acc = self.mmulacc_2d(acc, w16, x16)
            out_tile = acc[:, 0].copy()
            for col in range(tcol, n):  # K remainder (scalar)
                for i in range(N):
                    out_tile[i] = np.float32(
                        out_tile[i] + w[(row + i) * n + col] * x[col]
                    )
            xout[row : row + N] = out_tile

        for row in range(trow, d):  # boundary rows (scalar)
            xout[row] = self._dot_fp32(w, row, n, x, 1, 0)

    # ========== Helpers ==========

    def _stride_elems(self, stride_bytes: int) -> int:
        if stride_bytes % ELEM_BYTES != 0 or stride_bytes < 0:
            raise ValueError(
                f"stride {stride_bytes}B is not a whole number of fp32 elements"
            )
        return stride_bytes // ELEM_BYTES

    def _check_segment(self, buf: np.ndarray, off: int) -> None:
        if off < 0 or off + self.N > buf.size:
            raise IndexError("square access crosses the buffer boundary")

    @staticmethod
    def _dot_fp32(a, ai, K, b, bstride, bj) -> np.float32:
        acc = np.float32(0)
        for p in range(K):
            acc = np.float32(acc + a[ai * K + p] * b[p * bstride + bj])
        return acc


def _demo() -> None:
    rng = np.random.default_rng(0)
    tu = TileGEMM()  # N = amenlen = 8

    print("=" * 62)
    print("matmul: W @ x via mls.st / mls.tst(stride 0) / mconv / mmulacc")
    print("=" * 62)
    for d, n in [(16, 16), (24, 40), (33, 13)]:
        W = rng.standard_normal((d, n)).astype(np.float32)
        x = rng.standard_normal(n).astype(np.float32)
        out = np.zeros(d, dtype=np.float32)
        tu.matmul(out, x, W.reshape(-1), n, d)
        ref = W.astype(np.float64) @ x.astype(np.float64)
        err = np.max(np.abs(out - ref))
        print(
            f"  d={d:3d} n={n:3d}  max_err={err:.3e}  rel={err / max(1.0, np.abs(ref).max()):.3e}"
        )

    print("=" * 62)
    print("gemm: C = A @ B, spec portable triple loop (N x N squares)")
    print("=" * 62)
    for M, P, K in [(8, 8, 8), (16, 16, 16), (24, 16, 32), (13, 21, 9), (64, 48, 40)]:
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, P)).astype(np.float32)
        C = tu.gemm(A, B)
        ref = A.astype(np.float64) @ B.astype(np.float64)
        err = np.max(np.abs(C - ref))
        print(
            f"  M={M:3d} P={P:3d} K={K:3d}  max_err={err:.3e}  rel={err / max(1.0, np.abs(ref).max()):.3e}"
        )

    C = rng.standard_normal((16, 16)).astype(np.float32)
    A = rng.standard_normal((16, 16)).astype(np.float32)
    B = rng.standard_normal((16, 16)).astype(np.float32)
    got = tu.gemm(A, B, C.copy())
    ref = C.astype(np.float64) + A.astype(np.float64) @ B.astype(np.float64)
    print(f"  C += A @ B        max_err={np.max(np.abs(got - ref)):.3e}")


if __name__ == "__main__":
    _demo()
