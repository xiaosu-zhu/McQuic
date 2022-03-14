import sys
import os

__ENTRY_POINTS__ = {
    "mcquic": "mcquic.cli:entryPoint",
    "mcquic-train": "mcquic.train.cli:entryPoint",
    "mcquic-dataset": "mcquic.datasets.cli:entryPoint",
    "mcquic-validate": "mcquic.validate.cli:entryPoint"
}


template = """[options.entry_points]
console_scripts =
"""

with open(sys.argv[1], "a") as fp:
    fp.write(template)
    for key, value in __ENTRY_POINTS__.items():
        fp.write(f"    {key} = {value}{os.linesep}")
