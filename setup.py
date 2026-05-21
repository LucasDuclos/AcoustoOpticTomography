from setuptools import setup, find_packages
import sys
import subprocess
import os

# Default dependencies - only numpy, scipy, tqdm, matplotlib are required
# All other dependencies are optional
REQUIRED_DEPENDENCIES = [
    'numpy>=1.21.0',
    'scipy>=1.7.0',
    'tqdm>=4.60.0',
    'matplotlib>=3.4.0',
]

# Optional dependencies
OPTIONAL_DEPENDENCIES = {
    'gpu': [
        'cupy-cuda12x>=12.0.0',  # CuPy for GPU acceleration
    ],
    'acoustic': [
        'kwave>=0.3.5',  # For acoustic simulation (optional)
    ],
    'all': [
        'cupy-cuda12x>=12.0.0',
        'kwave>=0.3.5',
    ],
}

def check_cuda_available():
    """Check if CUDA is available on the system."""
    try:
        # Check for nvcc (NVIDIA CUDA Compiler)
        result = subprocess.run(['nvcc', '--version'], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True)
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_install_requires():
    """Get the install requirements based on command line arguments and environment."""
    # Check if user explicitly specified CPU mode via environment variable
    use_gpu = os.environ.get('AOT_BIOMAPS_CPU_ONLY', '').lower() != 'true'
    use_acoustic = os.environ.get('AOT_BIOMAPS_WITH_ACOUSTIC', '').lower() == 'true'
    
    # Also check command line arguments (for setup.py direct calls)
    if '--cpu' in sys.argv:
        use_gpu = False
        sys.argv.remove('--cpu')
    elif '--gpu' in sys.argv:
        use_gpu = True
        sys.argv.remove('--gpu')
    
    if '--with-acoustic' in sys.argv:
        use_acoustic = True
        sys.argv.remove('--with-acoustic')
    
    # Build install requirements
    install_requires = REQUIRED_DEPENDENCIES.copy()
    
    # Add GPU dependencies if requested or if CUDA is available
    if use_gpu:
        # Only add CuPy if CUDA is available on the system
        if check_cuda_available():
            install_requires.extend(OPTIONAL_DEPENDENCIES['gpu'])
            print("CUDA detected - including CuPy for GPU acceleration")
        else:
            print("Warning: GPU mode requested but CUDA not detected. Falling back to CPU-only installation.")
            print("Install CuPy manually if you want GPU support: pip install cupy-cuda12x")
    else:
        print("CPU-only installation requested - CuPy will not be installed")
    
    # Add acoustic dependencies if requested
    if use_acoustic:
        install_requires.extend(OPTIONAL_DEPENDENCIES['acoustic'])
        print("Including kWave for acoustic simulation")
    
    return install_requires

# Get the install requirements
extras_require = {
    'gpu': OPTIONAL_DEPENDENCIES['gpu'],
    'acoustic': OPTIONAL_DEPENDENCIES['acoustic'],
    'all': OPTIONAL_DEPENDENCIES['all'],
}

# Package data - include CUDA source file for compilation with CuPy
package_data = {
    'AOT_biomaps': [
        'AOT_Recon/AOT_biomaps_kernels.cu',
    ],
}

# Entry points for command line installation options
entry_points = {
    'console_scripts': [
        'aot-biomaps = AOT_biomaps.__main__:main',
    ],
}

setup(
    name='aot-biomaps',
    version='2.9.521',
    packages=find_packages(),
    package_dir={'': '.'},  # Look for packages in current directory
    include_package_data=True,
    package_data=package_data,
    
    # Core dependencies (always installed)
    install_requires=get_install_requires(),
    
    # Optional dependencies that can be installed with:
    # pip install aot-biomaps[gpu]
    # pip install aot-biomaps[acoustic]
    # pip install aot-biomaps[all]
    extras_require=extras_require,
    
    author='Lucas Duclos',
    author_email='lucas.duclos@universite-paris-saclay.fr',
    description='Acousto-Optic Tomography Reconstruction Library',
    long_description=open('README.md', encoding='utf-8').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/LucasDuclos/AcoustoOpticTomography',
    project_urls={
        'Documentation': 'https://github.com/LucasDuclos/AcoustoOpticTomography',
        'Source': 'https://github.com/LucasDuclos/AcoustoOpticTomography',
        'PyPI': 'https://pypi.org/project/aot-biomaps/',
    },
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    
    python_requires='>=3.8',
    
    # Entry points
    entry_points=entry_points,
)