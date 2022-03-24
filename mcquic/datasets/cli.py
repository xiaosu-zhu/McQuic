import json
import os
from random import shuffle
import shutil
import sys
import pathlib

import lmdb
from PIL import Image
import PIL
import click
from mcquic.train.utils import getRichProgress

from mcquic.utils import hashOfFile

_EXT = [".png", ".jpg", ".jpeg"]


def write(txn, i: bytes, path: str):
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


def getFilesFromDir(root, progress, strict: bool = False):
    files = findAllWithSize(root, _EXT)
    newFile = list()

    total = len(files)

    task = progress.add_task(f"[ Check ]", total=total, progress="0.00%", suffix=f"Scanning {total} images")

    for i, f in enumerate(files):
        try:
            a = Image.open(f[0])
        except PIL.UnidentifiedImageError:
            continue
        w, h = a.size
        if strict:
            # force images size > 512.
            if h < 512 or w < 512:
                continue
        newFile.append(f)

        progress.update(task, advance=1, progress=f"{(i + 1) / total * 100 :.2f}%")

    print(f"{len(newFile)} images meets requirement.")\

    progress.remove_task(task)

    return [x[0] for x in newFile]


def main(imageFolder: pathlib.Path, targetDir: pathlib.Path):
    shutil.rmtree(targetDir, ignore_errors=True)
    os.makedirs(targetDir, exist_ok=True)

    progress = getRichProgress()

    with progress:

        allFiles = getFilesFromDir(imageFolder, progress, True)

        shuffle(allFiles)

        env = lmdb.Environment(str(targetDir), subdir=True, map_size=int(1024 ** 4))

        total = len(allFiles)

        task = progress.add_task(f"[ Write ]", total=total, progress="0.00%", suffix=f"Writing {total} images")

        with env.begin(write=True) as txn:
            i = -1
            for i, f in enumerate(allFiles):
                write(txn, i.to_bytes(32, sys.byteorder), f)
                progress.update(task, advance=1, progress=f"{(i + 1) / total * 100 :.2f}%")
        env.close()

        progress.remove_task(task)


        dbFile = os.path.join(targetDir, "data.mdb")

        print("Calculate database hash...")

        hashResult = hashOfFile(dbFile, progress)

        # Create metadata needed for dataset
        with open(os.path.join(targetDir, "metadata.json"), "w") as fp:
            json.dump({
                "length": i + 1,
                "hash": hashResult
            }, fp)


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.argument("images", type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
@click.argument("output", type=click.Path(exists=False, file_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
def entryPoint(debug, quiet, path, images, output):
    """Create training set from `images` dir to `output` dir.

Args:

    images (str): All training images folder, allow sub-folders.

    output (str): Output dir to create training set.
    """
    main(images, output)
