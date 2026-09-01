import time
import numpy as np
from scipy import sparse


def to_sparse_csr(array, threshold: float = 0.0):
    """Zeroes out elements below a threshold and converts array to SciPy CSR format."""
    matrix = np.array(array, dtype=np.float64, copy=True)
    if threshold > 0.0:
        matrix[np.abs(matrix) <= threshold] = 0.0
    return sparse.csr_matrix(matrix)


def calculate_sparsity(array):
    """Calculates the percentage of zero values in a matrix."""
    total_elements = array.size
    if total_elements == 0:
        return 0.0
    zero_elements = total_elements - np.count_nonzero(array)
    return round((zero_elements / total_elements) * 100, 2)


def compare_memory(dense_array, sparse_matrix):
    """Computes exact RAM usage in bytes and KB for both dense and sparse representations."""
    dense_bytes = dense_array.nbytes

    # SciPy CSR memory = values array + column index array + row pointer array
    sparse_bytes = (
        sparse_matrix.data.nbytes
        + sparse_matrix.indices.nbytes
        + sparse_matrix.indptr.nbytes
    )

    savings_percent = 0.0
    if dense_bytes > 0:
        savings_percent = round((1 - (sparse_bytes / dense_bytes)) * 100, 2)

    return {
        'dense_bytes': dense_bytes,
        'sparse_bytes': sparse_bytes,
        'dense_kb': round(dense_bytes / 1024, 2),
        'sparse_kb': round(sparse_bytes / 1024, 2),
        'savings_percent': savings_percent
    }


def benchmark_multiplication(dense_array, sparse_matrix, runs: int = 10):
    """Times execution speed of matrix-vector multiplication for dense vs sparse formats."""
    cols = dense_array.shape[1]
    vector = np.random.rand(cols)

    # Benchmark Dense Array Multiplication
    start_dense = time.perf_counter()
    for _ in range(runs):
        _ = dense_array @ vector
    dense_time = (time.perf_counter() - start_dense) / runs

    # Benchmark Sparse Matrix Multiplication
    start_sparse = time.perf_counter()
    for _ in range(runs):
        _ = sparse_matrix @ vector
    sparse_time = (time.perf_counter() - start_sparse) / runs

    speedup = 0.0
    if sparse_time > 1e-9:
        speedup = round(dense_time / sparse_time, 2)
    else:
        speedup = 1.0

    return {
        'dense_time_ms': round(dense_time * 1000, 4),
        'sparse_time_ms': round(sparse_time * 1000, 4),
        'speedup_factor': speedup
    }

