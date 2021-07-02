import json
import os
from random import shuffle
import shutil

import tqdm
import lmdb
from PIL import Image

def write(txn, i: bytes, path: str):
    # fileName = os.path.basename(path)
    with open(path, "rb") as fp:
        txn.put(i, fp.read())

_EXT = [".png", ".jpg", ".jpeg"]

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

def sortBiggest(files, k):
    return sorted(files, key=lambda x: x[1], reverse=True)[:k]

def getMethodA(root):
    files = findAllWithSize(root, _EXT)
    newFile = list()
    for f in tqdm.tqdm(files):
        a = Image.open(f[0])
        h, w = a.size
        if h < 512 or w < 512:
            continue
        newFile.append(f)
    # files = sortBiggest(files, amount)
    print(len(newFile))
    return [x[0] for x in newFile]

def getMethodB(root):
    return [x[0] for x in findAllWithSize(root, _EXT)]

def main(targetDir):
    shutil.rmtree(targetDir, ignore_errors=True)
    os.makedirs(targetDir, exist_ok=True)
    env = lmdb.Environment(targetDir, subdir=True, map_size=1073741824 * 20)
    listA = [] # ["data/ImageNet/test", "data/ImageNet/val", "data/coco/train2014", "data/nus"]
    listB = ["data/clic/train", "data/DIV2K/train", "data/urban100", "data/manga109"]
    allFiles = list()
    for path in listA:
        allFiles.extend(getMethodA(path))
    for path in listB:
        allFiles.extend(getMethodB(path))
    os.makedirs(targetDir, exist_ok=True)
    shuffle(allFiles)
    with env.begin(write=True) as txn:
        for i, f in enumerate(tqdm.tqdm(allFiles)):
            write(txn, i.to_bytes(32, "big"), f)
    env.close()

    # Create metadata needed for dataset
    with open(os.path.join(targetDir, "metadata.json"), "w") as fp:
        json.dump({
            "length": i,
        }, fp)


if __name__ == "__main__":
    main("data/compression")
