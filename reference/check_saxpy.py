"""
Oracle for SAXPY: y = a*x + y.

Unlike matmul/matvec (BLAS-backed @ ), NumPy has no dedicated "saxpy"
call -- it's simple enough that a*x + y is itself already the
independent check (elementwise scalar multiply and add are NumPy
primitives we trust, not hand-rolled loops we're trying to verify).
"""
import numpy as np
from genmat_ref import gen_random

N = 1_000_000  # correctness-check size; VRAM is a non-issue for a
               # pure vector op, so this can be far bigger than the
               # N=4096 dense-matrix benchmark cap without any risk
A = 2.5        # fixed, reproducible scalar -- deliberately not 0 or 1,
               # since either of those would hide a broken axpy loop
               # (multiplying by 0 or 1 masks real bugs)

x = gen_random(N, 1, seed=60).flatten()
y = gen_random(N, 1, seed=61).flatten()

expected = A * x + y

np.save("expected_saxpy.npy", expected)
print(f"saxpy oracle: N={N}, a={A}, expected shape = {expected.shape}, "
      f"expected[0] = {expected[0]:.10f}")
