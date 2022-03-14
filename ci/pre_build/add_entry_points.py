import sys
import os
from .write_entry_points import __ENTRY_POINTS__


template = """[options.entry_points]
console_scripts =
"""

with open(sys.argv[1], "a") as fp:
    fp.write(template)
    for key, value in __ENTRY_POINTS__.items():
        fp.write(f"    {key} = {value}{os.linesep}")
