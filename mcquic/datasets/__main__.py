import json
import os
from random import shuffle
import shutil
import sys

import tqdm
import lmdb
from PIL import Image
import PIL
_EXT = [".png", ".jpg", ".jpeg"]

def write(txn, i: bytes, path: str):
    # fileName = os.path.basename(path)
    with open(path, "rb") as fp:
        txn.put(i, fp.read())

def findAllWithSize(dirPath, ext):
    files = list()
    filesAndDirs = os.listdir(dirPath)
    for f in filesAndDirs:
        if os.path.isdir(os.path.join(dirPath, f)):
            files.extend(findAllWithSize(os.path.join(dirPath, f), ext))
        elif os.path.splitext(f)[1].lower() in ext:
            f = os.path.join(dirPath, f)
            files.append((f, os.path.getsize(f)))
    return files

def getFilesFromDir(root, strict: bool = False):
    files = findAllWithSize(root, _EXT)
    newFile = list()
    for f in tqdm.tqdm(files, ncols=60, bar_format="{l_bar}{bar}| Search in %d images" % len(files)):
        try:
            a = Image.open(f[0])
        except PIL.UnidentifiedImageError:
            continue
        w, h = a.size
        if strict and (h < 512 or w < 512):
            continue
        newFile.append(f)
    print(f"{len(newFile)} images meets requirement.")
    return [x[0] for x in newFile]

FULL_DATASET = ["data/ImageNet/test", "data/ImageNet/val", "data/COCO/train2017", "data/clic/train", "data/DIV2K", "data/urban100", "data/manga109"]
EXTRA_DATSET = FULL_DATASET + ["data/paris", "data/oxford", "data/Flickr2K/Flickr2K_HR", "data/holiday"] # TOOOOOOOO BIG!: , "data/revisitop1m"]

NAMES = {
    "full": FULL_DATASET,
    "extra": EXTRA_DATSET
}

def main(targetDir, content):
    shutil.rmtree(targetDir, ignore_errors=True)
    os.makedirs(targetDir, exist_ok=True)
    lists = NAMES[content]
    allFiles = list()
    for path in lists:
        allFiles.extend(getFilesFromDir(path, True))
    os.makedirs(targetDir, exist_ok=True)
    shuffle(allFiles)
    env = lmdb.Environment(targetDir, subdir=True, map_size=int(1024 ** 4))
    with env.begin(write=True) as txn:
        i = -1
        for i, f in enumerate(tqdm.tqdm(allFiles, ncols=60, bar_format="{l_bar}{bar}| Write %d images..." % len(allFiles))):
            write(txn, i.to_bytes(32, sys.byteorder), f)
        # Create metadata needed for dataset
        with open(os.path.join(targetDir, "metadata.json"), "w") as fp:
            json.dump({
                "length": i + 1,
            }, fp)
    env.close()



if __name__ == "__main__":
    main("data/compression", "extra")
