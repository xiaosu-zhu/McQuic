import sys

with open(sys.argv[1]) as fp:
    version = fp.readlines()[0].strip().split()[-1].strip("\"")

with open(sys.argv[2], "r") as fp:
    requires = fp.readlines()

with open(sys.argv[2], "w") as fp:
    for line in requires:
        if line.startswith("mcquic"):
            line = f"mcquic>={version}"
        fp.write(line)
