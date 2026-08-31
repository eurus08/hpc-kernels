"""
Python reimplementation of common/genmat.c's xorshift32 PRNG.

Must match the C version bit-for-bit -- same seed, same sequence,
same values -- or the "oracle" isn't actually checking against the
same input the C kernels were given, which defeats the entire point
of having an independent reference.

See common/genmat.c for the canonical C version and the reasoning
for using xorshift32 instead of NumPy's own RNG here (NumPy's default
generator is a different algorithm entirely and would silently
produce a different sequence from the same seed).
"""
import numpy as np

MASK32 = 0xFFFFFFFF


def xorshift32_step(state: int) -> int:
    """One step of Marsaglia's xorshift32. `state` and the return value
    are both plain Python ints, masked to 32 bits after every shift --
    Python ints don't wrap at 32 bits on their own the way C's uint32_t
    does, so the mask is what makes this behave identically to the C
    version's automatic wraparound."""
    x = state & MASK32
    x ^= (x << 13) & MASK32
    x ^= (x >> 17)
    x ^= (x << 5) & MASK32
    return x & MASK32


def rand_uniform_sequence(seed: int, count: int) -> np.ndarray:
    """Reproduces genmat.c's rand_uniform(): count consecutive draws in
    [-1, 1), starting from `seed` (or 1, if seed is 0 -- xorshift32
    requires a nonzero state, same guard as the C side)."""
    state = seed if seed != 0 else 1
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        state = xorshift32_step(state)
        u = state / 4294967296.0  # 2**32
        out[i] = 2.0 * u - 1.0
    return out


def gen_random(rows: int, cols: int, seed: int) -> np.ndarray:
    """Matches genmat_random(): flat rows*cols draws, reshaped row-major
    (NumPy's default order) to mirror the C side's A[i*cols+j] indexing."""
    flat = rand_uniform_sequence(seed, rows * cols)
    return flat.reshape(rows, cols)


def gen_diag_dominant(n: int, seed: int) -> np.ndarray:
    """Matches genmat_diag_dominant() in common/genmat.c exactly,
    including draw order: the C version fills off-diagonal entries
    row-by-row, column-by-column (skipping the diagonal itself), THEN
    makes a second pass to set each diagonal entry. Reproducing that
    same two-pass order is what keeps the xorshift32 sequence aligned
    with the C side -- drawing values in a different order would still
    produce a valid diagonally-dominant matrix, just not the SAME one,
    breaking the reproducibility guarantee.
    """
    state = seed if seed != 0 else 1
    A = np.zeros((n, n), dtype=np.float64)

    # Pass 1: off-diagonal entries, row-major order, skipping i == j --
    # must match the C loop's exact iteration order.
    for i in range(n):
        for j in range(n):
            if i != j:
                state = xorshift32_step(state)
                u = state / 4294967296.0
                A[i, j] = 2.0 * u - 1.0

    # Pass 2: diagonal = that row's off-diagonal absolute sum + 1.0,
    # matching genmat.c's genmat_diag_dominant exactly.
    for i in range(n):
        row_sum = np.sum(np.abs(A[i, :]))  # A[i,i] is still 0 at this point
        A[i, i] = row_sum + 1.0

    return A
