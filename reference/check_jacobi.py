"""
Oracle for the Jacobi 2D 5-point stencil (D4).

Not a hand-rolled Python double loop -- that would just be a second
copy of the same per-point logic being tested. Instead this uses
NumPy's array slicing to compute all interior points for one whole
iteration as a single vectorized operation: grid[1:-1, 1:-1] read as
four shifted views (up/down/left/right neighbors) is independently
implemented from anything a C/CUDA loop does, even though the
underlying formula is identical.

Deterministic random interior + fixed zero boundary, rather than
reproducing any specific textbook heat-source scenario -- keeps this
oracle-checkable the same way every other kernel in this repo is,
via a fixed seed and a printed comparison value.
"""
import numpy as np
from genmat_ref import gen_random

N = 512       # interior grid dimension (matches transpose's correctness-check scale)
ITERS = 100   # fixed sweep count -- no convergence check, same as the plan's approach

# Full array includes a 1-cell-wide halo on every side for the fixed
# boundary -- shape (N+2, N+2), interior occupies indices 1..N.
grid = np.zeros((N + 2, N + 2), dtype=np.float64)
grid[1:N+1, 1:N+1] = gen_random(N, N, seed=70)
# Boundary (row/col 0 and row/col N+1) stays exactly 0.0, untouched
# for the whole run -- np.zeros() already gives us that.

for _ in range(ITERS):
    up    = grid[0:N,   1:N+1]
    down  = grid[2:N+2, 1:N+1]
    left  = grid[1:N+1, 0:N]
    right = grid[1:N+1, 2:N+2]
    new_interior = 0.25 * (up + down + left + right)
    grid[1:N+1, 1:N+1] = new_interior
    # Reading all four neighbor views BEFORE writing new_interior back
    # is what makes this a true Jacobi update -- every point's new
    # value comes entirely from the previous iteration's grid, never
    # a mix of old and already-updated neighbors.

# A fixed interior point for the printed sanity comparison -- same
# "print one value, compare by eye" convention as reduce_*.py/
# check_transpose.py. (256, 256) in 1-indexed interior coordinates.
check_i, check_j = 256, 256
print(f"jacobi oracle: N={N}, iters={ITERS}, "
      f"grid[{check_i}][{check_j}] = {grid[check_i, check_j]:.10f}")
