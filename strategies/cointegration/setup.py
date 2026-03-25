from setuptools import setup, Extension
import pybind11

ext = Extension(
    name="cointegration_engine",
    sources=[
        "binding.cpp",
        "MarketData.cpp",
        "MathStats.cpp",
        "PairScanner.cpp",
    ],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=["-std=c++17", "-O3", "-fopenmp"],
    extra_link_args=["-fopenmp"],
    language="cpp",
)

setup(
    name="cointegration_engine",
    ext_modules=[ext],
)
