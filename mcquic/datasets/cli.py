import json
import os
from random import shuffle
import shutil
import sys
import pathlib
import glob
import PIL.Image
import joblib
from joblib import Parallel, delayed
import contextlib

import webdataset as wds
from PIL import Image
import PIL
import click
from mcquic.train.utils import getRichProgress

from filelock import Timeout, FileLock

from mcquic.utils import hashOfFile

_EXT = [".png", ".jpg", ".jpeg"]


def write(sink: wds.ShardWriter, i: int, path: str):
    with open(path, "rb") as fp:
        sink.write(
            {
                '__key__': 'sample%08d' % i,
                'jpg': fp.read()
            }
        )


def findAllWithSize(dirPath, ext):
    files = list()
    for e in ext:
        beforeLen = len(files)
        print(f'Searching {e} in {dirPath} ...')
        finded = glob.glob(os.path.join(dirPath, '**', f'*{e.lower()}'))
        files.extend(finded)
        finded = glob.glob(os.path.join(dirPath, '**', f'*{e.upper()}'))
        files.extend(finded)
        print(f'Added {len(files) - beforeLen} files.')
    return files[:10000]


def _joblibValidateImage(path, strict):
    try:
        a = Image.open(path)
    except (PIL.UnidentifiedImageError, PIL.Image.DecompressionBombError):
        return None
    w, h = a.size
    if strict:
        # force images size > 512.
        if h < 512 or w < 512:
            return None
    return path


@contextlib.contextmanager
def rich_joblib(updateFn):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            updateFn(self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback


def getFilesFromDir(root, progress, strict: bool = False):
    files = findAllWithSize(root, _EXT)

    total = len(files)

    task = progress.add_task(f"[ Check ]", total=total, progress="0.00%", suffix=f"Scanning {total} images")

    class updateFn:
        def __init__(self):
            self._current = 0
        def __call__(self, advance):
            self._current = self._current + advance
            progress.update(task, advance=advance, progress=f"{self._current / total * 100 :.2f}%")

    with rich_joblib(updateFn()):
        result = Parallel(32)(delayed(_joblibValidateImage)(f, strict) for f in files)
    newFile = [x for x in result if x is not None]

    print(f"{len(newFile)} images ({len(newFile) / total * 100:0.2f}%) meets requirement.")

    progress.remove_task(task)

    return newFile


def main(imageFolder: pathlib.Path, targetDir: pathlib.Path):
    shutil.rmtree(targetDir, ignore_errors=True)
    os.makedirs(targetDir, exist_ok=True)

    progress = getRichProgress()

    with progress:

        allFiles = getFilesFromDir(imageFolder, progress, True)

        shuffle(allFiles)

        sink = wds.ShardWriter(os.path.join(targetDir, 'OpenImagesv7_%04d.tar.gz'), maxcount=10000000, maxsize=30000000000)

        total = len(allFiles)

        task = progress.add_task(f"[ Write ]", total=total, progress="0.00%", suffix=f"Writing {total} images")

        # class updateFn:
        #     def __init__(self):
        #         self._current = 0
        #     def __call__(self, advance):
        #         self._current = self._current + advance
        #         progress.update(task, advance=advance, progress=f"{self._current / total * 100 :.2f}%")
        i = -1
        for i, f in enumerate(allFiles):
            write(sink, i, f)
            progress.update(task, advance=1, progress=f"{(i + 1) / total * 100 :.2f}%")

        with open(os.path.join(targetDir, 'metadata.json'), 'w') as fp:
            json.dump({
                'length': len(allFiles)
            }, fp)

        progress.remove_task(task)


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
