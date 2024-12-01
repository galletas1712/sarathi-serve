from setuptools import setup, find_packages

setup(
    name="sarathi-serve",
    version="0.0.1",
    packages=find_packages(exclude=["benchmark_output", "sarathi_kernels"]),
)
