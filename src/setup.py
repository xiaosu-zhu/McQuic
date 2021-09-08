from pathlib import Path
import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

cwd = Path(__file__).resolve().parent
package_name = "mcqc"

os.environ["DEBUG_BUILD"] = "1"



def get_extensions():
    ext_modules = []
    rans32_dir = cwd / package_name / "cpp_exts/rans"
    extra_compile_args = ["-std=c++17"]
    if os.getenv("DEBUG_BUILD", None):
        extra_compile_args += ["-g"]
    else:
        extra_compile_args += ["-O3"]
    ext_modules.append(
        Pybind11Extension(
            name=f"{package_name}.ans32",
            sources=[str(s) for s in rans32_dir.glob("*.cpp")],
            language="c++",
            include_dirs=[rans32_dir],
            extra_compile_args=extra_compile_args,
        )
    )

    return ext_modules


setup(name="mcqc", version="0.9", packages=find_packages(), ext_modules=get_extensions(), cmdclass={"build_ext": build_ext})
