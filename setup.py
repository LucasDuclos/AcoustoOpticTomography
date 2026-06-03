from setuptools import setup, find_packages
import sys
import subprocess
import os
import platform

# Default dependencies
REQUIRED_DEPENDENCIES = [
    'numpy==1.26.4',
    'scipy==1.13.1',
    'tqdm==4.60.0',
    'matplotlib==3.9.2',
]

OPTIONAL_DEPENDENCIES = {
    'acoustic': ['k-wave-python==0.3.5'],
}

def is_windows():
    return platform.system() == 'Windows'

def check_cuda_available():
    """Check if CUDA is available (cross-platform)."""
    try:
        nvcc_cmd = 'nvcc.exe' if is_windows() else 'nvcc'
        result = subprocess.run(
            [nvcc_cmd, '--version'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, shell=is_windows()
        )
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.environ.get('CUDA_PATH') is not None

def detect_cuda_version():
    """Detect CUDA version (cross-platform)."""
    try:
        nvcc_cmd = 'nvcc.exe' if is_windows() else 'nvcc'
        result = subprocess.run(
            [nvcc_cmd, '--version'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, shell=is_windows()
        )
        import re
        match = re.search(r'release (\d+)\.(\d+)', result.stdout + result.stderr)
        if match:
            major = match.group(1)
            return f'cupy-cuda{major}x>=12.0.0'
        return 'cupy-cuda12x==13.6.0'
    except Exception:
        return None

def get_install_requires():
    """Get install requirements with OS-aware defaults."""
    use_gpu = os.environ.get('AOT_BIOMAPS_CPU_ONLY', '').lower() != 'true'
    use_acoustic = os.environ.get('AOT_BIOMAPS_WITHOUT_ACOUSTIC', '').lower() != 'true'

    if '--cpu' in sys.argv:
        use_gpu = False
        sys.argv.remove('--cpu')
    if '--without-acoustic' in sys.argv:
        use_acoustic = False
        sys.argv.remove('--without-acoustic')

    install_requires = REQUIRED_DEPENDENCIES.copy()

    if use_gpu and check_cuda_available():
        cupy_pkg = detect_cuda_version()
        if cupy_pkg:
            install_requires.append(cupy_pkg)
            print(f"CUDA detected - including {cupy_pkg} for GPU acceleration")

    if use_acoustic:
        install_requires.extend(OPTIONAL_DEPENDENCIES['acoustic'])
        
    return install_requires

setup(
    name='aot-biomaps',
    version='2.9.713',
    packages=find_packages(),
    package_dir={'': '.'},
    include_package_data=True,
    package_data={
        'AOT_biomaps': ['AOT_Recon/AOT_biomaps_kernels.cu'],
    },
    install_requires=get_install_requires(),
    extras_require={
        'acoustic': OPTIONAL_DEPENDENCIES['acoustic'],
    },
    author='Lucas Duclos',
    author_email='lucas.duclos@universite-paris-saclay.fr',
    description='Acousto-Optic Tomography Reconstruction Library',
    long_description=open('README.md', encoding='utf-8').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/LucasDuclos/AcoustoOpticTomography',
    python_requires='>=3.8',
    entry_points={
        'console_scripts': ['aot-biomaps = AOT_biomaps.__main__:main'],
    }
)










































































































































































