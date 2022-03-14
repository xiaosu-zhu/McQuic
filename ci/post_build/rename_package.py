import glob
import os
import sys
from pathlib import Path
import shutil

files = glob.glob(os.path.join(sys.argv[1], "*/*.tar.bz2"))


for f in files:
    platform = Path(f).parent.name
    fileName = Path(f).name
    dst = os.path.join(Path(f).parent, f"{platform}-{fileName}")
    shutil.move(f, dst)
    print(f"Renamed package as {dst}")
