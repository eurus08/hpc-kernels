"""
Oracle for Gaussian elimination: solve Ax = b for x, given A
diagonally dominant.

np.linalg.solve is LAPACK-backed -- independently implemented, not a
hand-rolled elimination loop -- same reasoning as using @ for matmul
and matvec.
"""
import numpy as np
from genmat_ref import gen_diag_dominant, gen_random

N = 256  # correctness-check size, not the N=4096 benchmark cap

A = gen_diag_dominant(N, seed=42)
b = gen_random(N, 1, seed=43).flatten()
expected = np.linalg.solve(A, b)

np.save("expected_gaussian.npy", expected)
print(f"gaussian oracle: N={N}, expected shape = {expected.shape}, "
      f"expected[0] = {expected[0]:.10f}")
