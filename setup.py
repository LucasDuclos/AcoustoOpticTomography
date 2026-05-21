from setuptools import setup, find_packages

setup(
    name='AOT_biomaps',
    version='2.9.517',
    packages=find_packages(),
    include_package_data=True,
    
    # --- FIX: Explicitly include the CUDA binary file (.cubin) ---
    package_data={
        # The key is the package name ('AOT_biomaps').
        # The value is a list of paths relative to the package directory.
        'AOT_biomaps': ['AOT_Recon/AOT_biomaps_kernels.cubin'],
    },
    # -------------------------------------------------------------
    
    install_requires=[
        'pycuda',
        'joblib>=1.3.2',
        'ipykernel==6.30.1',
        'k-wave-python==0.3.5',
        'setuptools==75.1.0',
        'pyyaml==6.0.2',
        'numba==0.61.2',
        'tqdm==4.67.1',
        'nvidia-ml-py3==7.352.0',
        'numpy==1.26.4',
        'torch==2.7.0',
        'scipy==1.13.1',
        'cupy-cuda12x==13.6.0',
        'scikit-image== 0.26.0',
    ],

    author='Lucas Duclos',
    author_email='lucas.duclos@universite-paris-saclay.fr',
    description='Acousto-Optic Tomography',
    url='https://github.com/LucasDuclos/AcoustoOpticTomography',
)




