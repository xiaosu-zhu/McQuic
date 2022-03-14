import os
import sys
import glob

for fileName in glob.glob(f"{sys.argv[1]}/Scripts/mcquic*-script.py"):
    with open(fileName) as fp:
        lines = fp.readlines()
    with open(fileName, "w") as fp:

        if lines[0].startswith("#!"):
            fp.write(lines[0].rstrip() + " -O" + os.linesep)
        else:
            fp.write(sys.argv[1] + "\\python -O" + os.linesep)

        fp.writelines(lines[1:])
