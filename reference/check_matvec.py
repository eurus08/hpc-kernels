"""
Oracle for matvec: y = A @ x.

Same reasoning as check_matmul.py -- NumPy's @ is independently
implemented (BLAS-backed), not a hand-rolled loop.
"""
import numpy as np
from genmat_ref import gen_random

N = 512  # correctness-check size, not the N=4096 benchmark cap

A = gen_random(N, N, seed=42)
x = gen_random(N, 1, seed=43).flatten()
expected = A @ x

np.save("expected_matvec.npy", expected)
print(f"matvec oracle: N={N}, expected shape = {expected.shape}, "
      f"expected[0] = {expected[0]:.10f}")
