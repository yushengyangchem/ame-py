"""End-to-end GEMM examples driven by the AMESimulator."""

import numpy as np

from ame_py.dtype import DType
from ame_py.simulator import AMESimulator


def run_gemm_example() -> AMESimulator:
    """
    Run a full GEMM example on the simulator:

        C[8x8] += A[8x8] × B[8x8]
    """
    print("\n" + "=" * 70)
    print("  AME GEMM EXAMPLE: C += A × B")
    print("  N = 8, U = 8, using int8 datatype")
    print("=" * 70 + "\n")

    sim = AMESimulator()
    sim.verbose = True

    # Configure data types: int8 operands, int32 accumulator.
    # M4 holds the int32 result; it must be aligned to the int32 group_size (4).
    sim.msettyp(0, DType.INT8)
    sim.msettyp(1, DType.INT8)
    sim.msettyp(4, DType.INT32)
    sim.asettyp(0, DType.INT32)

    print("\n--- Load data ---")

    A_data = np.arange(1, 65, dtype=np.int64).reshape(8, 8)
    B_data = np.arange(1, 65, dtype=np.int64).reshape(8, 8)
    sim.memory[0x1000] = A_data.flatten()
    sim.memory[0x2000] = B_data.flatten()

    sim.mls(0, 0x1000)
    sim.mls(1, 0x2000)

    print("\n--- Matrix multiply ---")

    sim.mzero_2d(0)
    sim.mmulacc_2d(0, 0, 1)

    print("\n--- Store result ---")

    sim.mmov_a_m(4, 0)
    sim.mss(4, 0x3000)
    sim.print_state()

    # Verify against a NumPy reference
    print("--- Verify ---")
    expected = A_data @ B_data
    actual = sim.M[4].reshape(8, 8)

    print(f"Expected (A×B):\n{expected}")
    print(f"Actual (M4):\n{actual}")

    if np.allclose(expected, actual):
        print("[OK] Results match!")
    else:
        diff = np.abs(expected - actual)
        print(f"[WARN] Mismatch, max error: {np.max(diff)}")

    return sim


def run_portable_gemm_example() -> AMESimulator:
    """Run a portable GEMM that reads N and U from CSRs at runtime."""
    print("\n" + "=" * 70)
    print("  PORTABLE GEMM EXAMPLE")
    print("  Reads N and U from CSRs at runtime")
    print("=" * 70 + "\n")

    sim = AMESimulator()
    sim.verbose = True

    # On real hardware these would be csrr reads
    N = sim.amenlen
    U = sim.ameudsz

    print(f"Read from CSRs: N={N}, U={U}")

    sim.msettyp(0, DType.INT8)
    sim.msettyp(1, DType.INT8)
    sim.asettyp(0, DType.INT32)

    A_data = np.random.randint(0, 20, size=(N, N))
    B_data = np.random.randint(0, 20, size=(N, N))

    sim.memory[0x1000] = A_data.flatten()
    sim.memory[0x2000] = B_data.flatten()

    sim.mls(0, 0x1000)
    sim.mls(1, 0x2000)
    sim.mzero_2d(0)
    sim.mmulacc_2d(0, 0, 1)

    expected = A_data @ B_data
    actual = sim.Acc[0].reshape(N, N)

    print("\n--- Result ---")
    print(f"Expected (A×B):\n{expected}")
    print(f"Actual (Acc0):\n{actual}")

    if np.allclose(expected, actual):
        print("[OK] Results match!")
    else:
        print("[WARN] Mismatch")

    return sim


def main() -> None:
    print("\n" + "█" * 70)
    print("  AME SIMULATOR")
    print("  Attached Matrix Extension - Python Implementation")
    print("█" * 70)

    run_gemm_example()
    run_portable_gemm_example()

    print("\n" + "█" * 70)
    print("  Done!")
    print(
        "  Instructions: msettyp, asettyp, mls, mls.rm, mss, mmulacc.2d, mmov.a.m, mzero.2d"
    )
    print("  CSRs: amestatus, amenlen, ameudsz")
    print("█" * 70)


if __name__ == "__main__":
    main()
