from posix import listdir
import lmdb
import os
import tqdm
import shutil


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


def getImageNet(imageNetRoot):
    valSet = os.path.join(imageNetRoot, "val")
    files = findAllWithSize(valSet, _EXT)
    files = sortBiggest(files, 8000)
    return [x[0] for x in files]

def getCLIC(clicRoot):
    return [x[0] for x in findAllWithSize(os.path.join(clicRoot, "train"), _EXT)]


def main(targetDir):
    shutil.rmtree(targetDir, ignore_errors=True)
    os.makedirs(targetDir, exist_ok=True)
    env = lmdb.Environment(targetDir, subdir=True, map_size=1073741824 * 20)
    imgFiles = getImageNet("data/ImageNet")
    clicFiles = getCLIC("data/clic")
    print("Find %d largest files in ImageNet" % len(imgFiles))
    print("Find %d train files in CLIC" % len(clicFiles))
    os.makedirs(targetDir, exist_ok=True)
    with env.begin(write=True) as txn:
        i = 0
        for f in tqdm.tqdm(imgFiles):

            write(txn, i.to_bytes(32, "big"), f)
            i += 1
        j = i
        for f in tqdm.tqdm(clicFiles):
            write(txn, i.to_bytes(32, "big"), f)
            i += 1

        txn.put(b"length", i.to_bytes(32, "big"))
        txn.put(b"imageNetLength", j.to_bytes(32, "big"))
        txn.put(b"clicLength", (i - j).to_bytes(32, "big"))
    env.close()


if __name__ == "__main__":
    main("data/compression")
