from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name = "GenEO",
    version="0.1",
    packages=find_packages(),
    ext_modules = cythonize('GenEO/matelem_cython.pyx'),
)