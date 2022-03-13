import os
import sys

def getBuildNumber(releaseVersion):
    main, major, minor = releaseVersion.split(".")
    buildNumber = 0x010000 * int(main) + 0x0100 * int(major) + 0x01 * int(minor)
    return buildNumber

def updateBuildNumber(fileName, releaseVersion):
    with open(fileName) as fp:
        lines = fp.readlines()

    with open(fileName, "w") as fp:
        for line in lines:
            if line.startswith("  number:"):
                fp.write(f"  number: {getBuildNumber(releaseVersion)}{os.linesep}")
            else:
                fp.write(line)


if __name__ == "__main__":
    updateBuildNumber("./meta.yaml", sys.argv[1])
