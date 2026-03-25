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
    include_dirs=[
        pybind11.get_include(),
        "eigen-5.0.0",
    ],
    extra_compile_args=["/std:c++17", "/O2", "/EHsc"],
    extra_link_args=[],
    language="cpp",
)

setup(
    name="cointegration_engine",
    ext_modules=[ext],
)
