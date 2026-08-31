"""
Oracle for matmul: C = A @ B.

NumPy's `@` operator calls into BLAS under the hood (typically OpenBLAS
or MKL) -- a heavily-optimized, independently-implemented matmul with
decades of correctness scrutiny behind it. That independence is exactly
what makes it trustworthy as ground truth here.
"""
import numpy as np
from genmat_ref import gen_random

N = 256  # small on purpose -- oracle just needs correctness, not the
         # N=4096 performance-benchmark size (see check_reduction.py note)

A = gen_random(N, N, seed=42)
B = gen_random(N, N, seed=43)  # different seed -- A and B must be
                                # independent draws, not the same matrix
expected = A @ B

np.save("expected_matmul.npy", expected)
print(f"matmul oracle: N={N}, expected shape = {expected.shape}, "
      f"expected[0,0] = {expected[0,0]:.10f}")
