from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        name="timetree",
        sources=["src/timetree_bindings.cpp"],
        include_dirs=["src"],
        cxx_std=17,
    )
]

setup(
    name="plaster-timetree",
    version="0.0.1",
    description="Pybind11 bindings for TimeTree (AVL timestamp index)",
    author="brown-ivl",
    packages=[],  # extension-only
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
