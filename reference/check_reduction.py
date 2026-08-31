"""
Oracle for the reduction kernel: sum of a vector.

Ground truth comes from NumPy's own sum, never a hand-written Python
loop -- the whole point of an oracle is an independent implementation,
and a hand-rolled Python sum would just be a second copy of the same
"map-then-reduce" logic being tested, not a real check on it.
"""
import numpy as np
from genmat_ref import gen_random

N = 1024
x = gen_random(N, 1, seed=42).flatten()
expected = np.sum(x)

np.save("expected_reduction.npy", expected)
print(f"reduction oracle: N={N}, expected sum = {expected:.10f}")
