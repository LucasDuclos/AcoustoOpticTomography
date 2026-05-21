#!/usr/bin/env python3
"""
install.py

User-friendly installation script for AOT_biomaps.
Allows users to choose between CPU and GPU installations.
"""

import subprocess
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Install AOT_biomaps - Acousto-Optic Tomography Reconstruction Library',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install with GPU support (default if CUDA available)
  python install.py

  # Install CPU only
  python install.py --cpu

  # Install with GPU explicitly
  python install.py --gpu

  # Install with acoustic simulation support
  python install.py --with-acoustic

  # Install everything (GPU + acoustic)
  python install.py --all
        """
    )
    
    parser.add_argument('--cpu', action='store_true', 
                       help='Install CPU only (no CuPy)')
    parser.add_argument('--gpu', action='store_true', 
                       help='Install with GPU support (requires CUDA)')
    parser.add_argument('--with-acoustic', action='store_true', 
                       help='Include kWave for acoustic simulation')
    parser.add_argument('--all', action='store_true', 
                       help='Install all optional dependencies')
    parser.add_argument('--dev', action='store_true', 
                       help='Install in development mode (editable)')
    parser.add_argument('--upgrade', action='store_true', 
                       help='Upgrade existing dependencies')
    
    args = parser.parse_args()
    
    # Determine installation mode
    install_cmd = []
    
    # Handle dependencies first (before building command)
    if args.cpu:
        # CPU only installation
        os.environ['AOT_BIOMAPS_CPU_ONLY'] = 'true'
        print("CPU-only installation...")
    elif args.gpu:
        # GPU installation explicitly
        os.environ['AOT_BIOMAPS_CPU_ONLY'] = 'false'
        print("Installing with GPU support...")
    
    if args.with_acoustic or args.all:
        os.environ['AOT_BIOMAPS_WITH_ACOUSTIC'] = 'true'
        print("Including kWave for acoustic simulation...")
    
    # Build installation command
    if args.dev:
        install_cmd.extend(['develop'])  # Editable mode
    else:
        install_cmd.append('install')
    
    if args.upgrade:
        install_cmd.insert(0, '--upgrade')
    
    # Execute installation
    cmd = [sys.executable, 'setup.py'] + install_cmd
    
    print(f"Executing: {' '.join(cmd)}")
    print("-" * 50)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Warnings:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    
    if result.returncode != 0:
        print(f"Installation failed (code: {result.returncode})")
        sys.exit(result.returncode)
    
    print("-" * 50)
    print("Installation completed successfully!")
    
    # Show test instructions
    print("\nTo test the installation:")
    print("  python -m AOT_biomaps test")
    
    print("\nTo see information:")
    print("  python -m AOT_biomaps info --dependencies")

if __name__ == '__main__':
    main()