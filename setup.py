from setuptools import setup, find_packages

setup(
    name='AOT_biomaps',
    version='0.8.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'k-wave-python',
    ],
    author='Lucas Duclos',
    author_email='lucas.duclos@universite-paris-saclay.fr',
    description='Acousto-Optic Tomography',
    long_description=open('C:/Users/duclos/Desktop/AcoustoOpticTomography/TomographyAcoustoOptic/README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/LucasDuclos/AcoustoOpticTomography',
)
