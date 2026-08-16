import numpy as np

from ame_py.examples import run_gemm_example


def test_gemm_example_matches_numpy(capsys):
    sim = run_gemm_example()
    capsys.readouterr()

    expected = sim.memory[0x1000].reshape(8, 8) @ sim.memory[0x2000].reshape(8, 8)
    actual = sim.memory[0x3000].reshape(8, 8)
    np.testing.assert_array_equal(actual, expected)
