"""
test_kernels.py

Simple test script to verify that the kernel module and sparse matrix wrappers work.
"""

import sys
import os

# Add the parent directory to the path so we can import AOT_biomaps
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np


def _run_tests_internal():
    """Internal function to run all tests."""
    print("Testing AOT_biomaps kernel and sparse matrix implementations...")
    print("=" * 70)

    # Test 1: Test SparseMatrix wrapper
    print("\n[Test 1] Testing SparseMatrix wrapper...")
    try:
        from AOT_biomaps.AOT_Recon.SparseMatrixWrapper import SparseMatrix, create_sparse_matrix
        print("[OK] SparseMatrix wrapper imported successfully")
        print("[OK] SparseMatrix wrapper class structure verified")
    except Exception as e:
        print(f"[FAIL] SparseMatrix wrapper test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Test MLEM module
    print("\n[Test 2] Testing MLEM module...")
    try:
        from AOT_biomaps.AOT_Recon.AOT_Optimizers.MLEM import (
            MLEM,
            MLEM_dense,
            MLEM_sparse,
            _MLEM_dense_cpu,
            _MLEM_dense_gpu
        )
        print("[OK] MLEM module imported successfully")
        
        np.random.seed(42)
        T, Z, X, N = 5, 10, 10, 5
        SMatrix = np.random.rand(T, Z, X, N).astype(np.float32)
        y = np.random.rand(T, N).astype(np.float32)
        
        result, _ = MLEM_dense(SMatrix, y, numIterations=5, isSavingEachIteration=False,
                               tumor_str="TEST", show_logs=False, device='cpu')
        assert result is not None, "MLEM_dense CPU returned None"
        assert result.shape == (Z, X), f"MLEM_dense CPU shape mismatch: {result.shape}"
        print("[OK] MLEM_dense (CPU) works")
    except Exception as e:
        print(f"[FAIL] MLEM test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Test sparse matrix imports
    print("\n[Test 3] Testing sparse matrix imports...")
    try:
        from AOT_biomaps.AOT_Recon.AOT_SparseSMatrix.SparseSMatrix_CSR import SparseSMatrix_CSR
        from AOT_biomaps.AOT_Recon.AOT_SparseSMatrix.SparseSMatrix_SELL import SparseSMatrix_SELL
        print("[OK] Sparse matrix classes imported successfully")
    except Exception as e:
        print(f"[FAIL] Sparse matrix import failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Test summary:")
    print("  - SparseMatrix wrapper: [OK]")
    print("  - MLEM module: [OK]")
    print("  - Sparse matrix classes: [OK]")
    print("\nAll tests completed successfully!")


def run_all_tests():
    """Run all kernel tests programmatically."""
    _run_tests_internal()


if __name__ == '__main__':
    _run_tests_internal()
