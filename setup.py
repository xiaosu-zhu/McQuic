import os
from pathlib import Path
from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext, ParallelCompile

ParallelCompile("NPY_NUM_BUILD_JOBS").install()

CWD = Path("./")
PKG_NAME = "mcquic"

# https://github.com/InterDigitalInc/CompressAI/blob/master/setup.py
# Copyright (c) 2021-2022, InterDigital Communications, Inc
def get_extensions():
    ext_dirs = CWD / "third_party/cpp_exts"
    ext_modules = []

    # Add rANS module
    rans_lib_dir = CWD / "third_party/ryg_rans"

    if os.name == "nt":
        compiler_args = ["/std:c++17", "/O2", "/GL", "/MP8"]
    else:
        compiler_args = ["-std=c++17", "-O3"]
    ext_modules.append(
        Pybind11Extension(
            name=f"{PKG_NAME}.rans",
            sources=[str(s) for s in ext_dirs.glob("*.cpp")],
            language="c++",
            include_dirs=[rans_lib_dir, ext_dirs],
            extra_compile_args=compiler_args,
        )
    )

    return ext_modules


distribution = setup(
    ext_modules = get_extensions(),
    cmdclass = {
        "build_ext": build_ext
    }
)
