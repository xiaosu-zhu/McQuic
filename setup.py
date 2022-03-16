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
    ext_dirs = CWD / "third_party/CompressAI/cpp_exts"
    ext_modules = []

    # Add rANS module
    rans_lib_dir = CWD / "third_party/CompressAI/ryg_rans"

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


setupArgs = {
    "ext_modules": get_extensions(),
    "cmdclass": {
        "build_ext": build_ext
    }
}

if os.getenv("PYPI_BUILDING", "") != "":
    install_requires = [
        "torch<2",
        "torchvision",
        "tqdm",
        "tensorboard<3",
        "rich<11",
        "lmdb<2",
        "pyyaml<7",
        "marshmallow<4",
        "click<9",
        "vlutils",
        "msgpack-python<2"
    ]
    setupArgs.update({
        "install_requires": install_requires
    })


setup(**setupArgs)
