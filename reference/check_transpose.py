"""
Oracle for transpose: B = A^T.

Unlike matmul/matvec/gaussian, transpose is pure data movement -- no
arithmetic, no floating-point rounding at all. A correct C
implementation should match this oracle to EXACTLY zero relative
error, not just "small" -- if verify.c ever reports a nonzero error
here, that's a real indexing bug, not FP noise, and should be treated
that way.
"""
import numpy as np
from genmat_ref import gen_random

N = 512  # correctness-check size, not the N=4096 benchmark cap

A = gen_random(N, N, seed=42)
expected = A.T.copy()  # .copy(): NumPy's .T is a view, not new data --
                        # .copy() forces an actual transposed array in
                        # memory, matching what a real kernel produces

np.save("expected_transpose.npy", expected)
print(f"transpose oracle: N={N}, expected shape = {expected.shape}, "
      f"expected[0,1] = {expected[0,1]:.10f}, A[1,0] = {A[1,0]:.10f} (should match)")
