"""
Main ``np-nurbs`` library
"""
# import time

from numpy.typing import NDArray
import numpy as np
from scipy.special import comb


def generate_cpu_coefficient_matrix(degree: int) -> NDArray[np.int64]: 
    """
    Generates a coefficient matrix used to apply the combination
    function for a given Bernstein polynomial degree

    Parameters
    ----------
    degree: int
        Polynomial degree

    Returns
    -------
    NDArray[np.int64]
        Square array of size ``degree + 1``
    """
    matrix_size = degree + 1

    # 1. Initialize the index arrays
    k = np.arange(matrix_size).reshape(-1, 1)
    i = np.arange(matrix_size)

    # 2. Calculate the difference used in the formula
    diff = degree - k - i
    
    # 3. Create a safety mask for valid indices
    mask = (diff >= 0)
    
    # 4. Use np.maximum to prevent negative exponents for (-1)**diff
    # This avoids the ValueError while preserving correct parity for 
    # valid cells
    safe_diff = np.maximum(diff, 0)
    
    # 5. Calculate components with NumPy's vectorized broadcasting
    signs = (-1)**safe_diff
    c1 = comb(degree, i)
    c2 = comb(degree - i, safe_diff)
    
    # 6. Combine and apply mask
    M = signs * c1 * c2
    M = np.where(mask, M, 0)
    return np.where(mask, M, 0)


# start_time = time.perf_counter()
coefficient_matrices: dict[int, NDArray[np.int64]] = {
    i: generate_cpu_coefficient_matrix(i) for i in range(32)
}
# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"CPU coefficient matrix initialization time: {elapsed_time:.6f} seconds")


# Delayed import of all functions so that each function can import the
# just instantiated hashmap of coefficient matrices
from .bezier import *
from .rational_bezier import *

