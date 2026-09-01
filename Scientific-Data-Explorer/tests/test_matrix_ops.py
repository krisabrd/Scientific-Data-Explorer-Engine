import numpy as np

from backend.src.matrix_ops import calculate_sparsity, compare_memory, to_sparse_csr


def test_calculate_sparsity_returns_float():
    array = np.array([[1.0, 0.0], [0.0, 2.0]])

    value = calculate_sparsity(array)

    assert isinstance(value, float)
    assert abs(value - 50.0) < 1e-9


def test_to_sparse_csr_and_memory_metrics():
    array = np.array([[1.0, 0.0], [0.0, 2.0]])

    sparse_matrix = to_sparse_csr(array, threshold=0.0)
    mem_stats = compare_memory(array, sparse_matrix)

    assert sparse_matrix.shape == (2, 2)
    assert mem_stats["dense_kb"] >= 0.0
    assert mem_stats["sparse_kb"] >= 0.0
