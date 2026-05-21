#!/usr/bin/env python3
"""
AOT_biomaps CLI Entry Point

Provides command-line interface for testing and information about the library.
"""

import argparse
import sys
import traceback


def test_imports():
    """Test that all main imports work correctly."""
    print("Testing AOT_biomaps imports...")
    
    try:
        import AOT_biomaps
        print(f"  [OK] AOT_biomaps version: {AOT_biomaps.__version__}")
    except Exception as e:
        print(f"  [FAIL] AOT_biomaps: {e}")
        return False
    
    try:
        from AOT_biomaps.AOT_Recon import AOT_Kernels
        print(f"  [OK] AOT_Kernels")
    except Exception as e:
        print(f"  [FAIL] AOT_Kernels: {e}")
        return False
    
    try:
        from AOT_biomaps.AOT_Recon import SparseMatrixWrapper
        print(f"  [OK] SparseMatrixWrapper")
    except Exception as e:
        print(f"  [FAIL] SparseMatrixWrapper: {e}")
        return False
    
    try:
        from AOT_biomaps.AOT_Recon.AOT_Optimizers import MLEM
        print(f"  [OK] MLEM")
    except Exception as e:
        print(f"  [FAIL] MLEM: {e}")
        return False
    
    try:
        from AOT_biomaps.AOT_Recon.AOT_Optimizers import PDHG
        print(f"  [OK] PDHG")
    except Exception as e:
        print(f"  [FAIL] PDHG: {e}")
        return False
    
    try:
        from AOT_biomaps.AOT_Recon.AOT_Optimizers import LS
        print(f"  [OK] LS")
    except Exception as e:
        print(f"  [FAIL] LS: {e}")
        return False
    
    try:
        from AOT_biomaps.AOT_Recon.AOT_Optimizers import DEPIERRO
        print(f"  [OK] DEPIERRO")
    except Exception as e:
        print(f"  [FAIL] DEPIERRO: {e}")
        return False
    
    try:
        from AOT_biomaps.AOT_Recon.AOT_Optimizers import MAPEM
        print(f"  [OK] MAPEM")
    except Exception as e:
        print(f"  [FAIL] MAPEM: {e}")
        return False
    
    try:
        from AOT_biomaps.AOT_Recon.AOT_Optimizers import LBFGS
        print(f"  [OK] LBFGS")
    except Exception as e:
        print(f"  [FAIL] LBFGS: {e}")
        return False
    
    try:
        from AOT_biomaps.AOT_Recon import ReconTools
        print(f"  [OK] ReconTools")
    except Exception as e:
        print(f"  [FAIL] ReconTools: {e}")
        return False
    
    try:
        from AOT_biomaps.AOT_Experiment import Tomography
        print(f"  [OK] Tomography")
    except Exception as e:
        print(f"  [FAIL] Tomography: {e}")
        return False
    
    try:
        from AOT_biomaps.AOT_Acoustic import AcousticTools
        print(f"  [OK] AcousticTools")
    except Exception as e:
        print(f"  [FAIL] AcousticTools: {e}")
        return False
    
    try:
        from AOT_biomaps.AOT_Medium import HomogeneousMedium, PVAMedium, BubbleMedium
        print(f"  [OK] Medium classes")
    except Exception as e:
        print(f"  [FAIL] Medium classes: {e}")
        return False
    
    print("\nAll imports successful!")
    return True


def show_info(args):
    """Show information about the library."""
    import AOT_biomaps
    
    print(f"AOT_biomaps version: {AOT_biomaps.__version__}")
    print(f"Author: Lucas Duclos")
    print(f"Description: {AOT_biomaps.__doc__ or 'Acousto-Optic Tomography Reconstruction Library'}")
    
    if args.dependencies:
        print("\nDependencies:")
        try:
            import pkg_resources
            dist = pkg_resources.get_distribution('aot-biomaps')
            deps = [str(r) for r in dist.requires()]
            for dep in deps:
                print(f"  - {dep}")
        except Exception as e:
            print(f"  Could not retrieve dependencies: {e}")
        
        # Check optional dependencies
        print("\nOptional dependencies status:")
        
        try:
            import cupy
            print(f"  [INSTALLED] CuPy: {cupy.__version__}")
        except ImportError:
            print(f"  [NOT INSTALLED] CuPy (for GPU acceleration)")
        
        try:
            import kwave
            print(f"  [INSTALLED] kWave: {kwave.__version__}")
        except ImportError:
            print(f"  [NOT INSTALLED] kWave (for acoustic simulation)")
        
        try:
            import matplotlib
            print(f"  [INSTALLED] matplotlib: {matplotlib.__version__}")
        except ImportError:
            print(f"  [NOT INSTALLED] matplotlib (for visualization)")


def run_tests(args):
    """Run kernel tests."""
    try:
        from AOT_biomaps.AOT_Recon.test_kernels import run_all_tests
        run_all_tests()
    except Exception as e:
        print(f"Error running tests: {e}")
        traceback.print_exc()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='AOT_biomaps - Acousto-Optic Tomography Reconstruction Library CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m AOT_biomaps test        # Test all imports
  python -m AOT_biomaps info        # Show library info
  python -m AOT_biomaps info --dependencies  # Show dependencies
  python -m AOT_biomaps test-kernels # Run kernel tests
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test all imports')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show library information')
    info_parser.add_argument('--dependencies', action='store_true', 
                           help='Show dependency information')
    
    # Test-kernels command
    kernels_parser = subparsers.add_parser('test-kernels', 
                                          help='Run kernel tests')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == 'test':
        success = test_imports()
        sys.exit(0 if success else 1)
    elif args.command == 'info':
        show_info(args)
    elif args.command == 'test-kernels':
        run_tests(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
