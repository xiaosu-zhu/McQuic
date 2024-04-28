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
import datetime
from tqdm import tqdm
# from multiprocessing import Queue

import webdataset as wds
from PIL import Image
import PIL
import click
from mcquic.train.utils import getRichProgress

from mcquic.utils import hashOfFile

_EXT = [".png", ".jpg", ".jpeg"]

FILENAME = 'mcquic_DATA_%05d.tar.gz'


from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False


def write(sink: wds.ShardWriter, i: int, path: str, text: str):
    with open(path, "rb") as fp:
        sink.write(
            {
                '__key__': 'sample%08d' % i,
                'jpg': fp.read(),
                'txt': text
            }
        )


def findAllWithSize(dirPath, ext):
    files = list()
    for e in ext:
        beforeLen = len(files)
        print(f'Searching {e} in {dirPath} ...')
        finded = glob.glob(os.path.join(dirPath, '**', f'*{e.lower()}'), recursive=True)
        files.extend(finded)
        finded = glob.glob(os.path.join(dirPath, '**', f'*{e.upper()}'), recursive=True)
        files.extend(finded)
        print(f'Added {len(files) - beforeLen} files.')
    return files


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
    # check corresponding text
    if os.path.exists(pathlib.Path(path).with_suffix('.txt')):
        return path
    return None


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


def createwdsSingle(rank: int, start: int, files, targetDir):
    targetDir = os.path.join(targetDir, f'split{rank:03d}')
    shutil.rmtree(targetDir, ignore_errors=True)
    os.makedirs(targetDir, exist_ok=True)
    # 00000 ~ 05000
    sink = wds.ShardWriter(os.path.join(targetDir, FILENAME), maxcount=1000000)
    for i, f in enumerate(tqdm(files)):
        write(sink, start + i, f, pathlib.Path(f).with_suffix('.txt').read_text())


def combineAllSplits(targetDir, progress):
    allSplits = sorted(glob.glob(os.path.join(targetDir, 'split*')))

    task = progress.add_task(f"[ Check ]", total=len(allSplits), progress="0.00%", suffix=f"Moving {len(allSplits)} splits...")

    current = 0
    for i, split in enumerate(allSplits):
        allTars = sorted(glob.glob(os.path.join(split, '*.tar.gz')))
        for tar in allTars:
            shutil.move(tar, os.path.join(targetDir, FILENAME % current))
            print(f'Moved {tar} to {os.path.join(targetDir, FILENAME % current)}')
            current += 1
        progress.update(task, advance=1, progress=f"{i / len(allSplits) * 100 :.2f}%")
    for split in allSplits:
        shutil.rmtree(split)
    return

def main(imageFolder: pathlib.Path, targetDir: pathlib.Path, parallel: int = 32):
    shutil.rmtree(targetDir, ignore_errors=True)
    os.makedirs(targetDir, exist_ok=True)

    progress = getRichProgress()

    with progress:
        allFiles = getFilesFromDir(imageFolder, progress, True)

        shuffle(allFiles)

        lenSplit = len(allFiles) // parallel

        fileGroups, starts = list(), list()

        start = 0
        for i in range(parallel):
            fileGroups.append(allFiles[i * lenSplit:(i + 1) * lenSplit])
            starts.append(start)
            start += len(fileGroups[-1])

        Parallel(parallel)(delayed(createwdsSingle)(rank, start, files, targetDir) for rank, (start, files) in enumerate(zip(starts, fileGroups)))

        # 00000 ~ 05000
        # sink = wds.ShardWriter(os.path.join(targetDir, 'OpenImagesv7_%05d.tar.gz'), maxcount=1000000)

        # total = len(allFiles)

        # task = progress.add_task(f"[ Write ]", total=total, progress="0.00%", suffix=f"Writing {total} images")

        # class updateFn:
        #     def __init__(self):
        #         self._current = 0
        #     def __call__(self, advance):
        #         self._current = self._current + advance
        #         progress.update(task, advance=advance, progress=f"{self._current / total * 100 :.2f}%")
        # i = -1
        # for i, f in enumerate(allFiles):
        #     write(sink, i, f)
        #     progress.update(task, advance=1, progress=f"{(i + 1) / total * 100 :.2f}%")

        combineAllSplits(targetDir, progress)

    with open(os.path.join(targetDir, 'metadata.json'), 'w') as fp:
        json.dump({
            'length': len(allFiles),
            'from': str(imageFolder),
            'creation': datetime.datetime.now().strftime('%Y-%m-%d, %H:%M:%S')
        }, fp)

    # progress.remove_task(task)


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.option("-j", "--jobs", type=int, default=32, show_default=True, help="Parallelized processing jobs.")
@click.argument("images", type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
@click.argument("output", type=click.Path(exists=False, file_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
def entryPoint(debug, quiet, jobs, images, output):
    """Create training set from `images` dir to `output` dir.

Args:

    images (str): All training images folder, allow sub-folders.

    output (str): Output dir to create training set.
    """
    main(images, output, jobs)
