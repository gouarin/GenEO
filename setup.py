from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name = "elasticity",
    version="0.1",
    packages=find_packages(),
    ext_modules = cythonize('elasticity/matelem_cython.pyx'),
)