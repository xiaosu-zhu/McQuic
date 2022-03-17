from typing import Callable, Optional, Tuple, Callable, List, Union, cast
import os
import json
import sys
from pathlib import Path

import lmdb
import torch
from torch import Tensor
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode, decode_image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from vlutils.runtime import relativePath
from vlutils.types import StrPath


__all__ = [
    "Basic",
    "BasicLMDB"
]


def _hasFileAllowedExtension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def _makeDataset(directory: StrPath, extensions: Optional[Tuple[str, ...]] = None, is_valid_file: Optional[Callable[[str], bool]] = None,) -> List[str]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    def _validFileWrapper(x):
        return _hasFileAllowedExtension(x, cast(Tuple[str, ...], extensions))
    if extensions is not None:
        is_valid_file = _validFileWrapper
    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                instances.append(path)
    return instances


class Basic(VisionDataset):
    """A Basic dataset that reads all images from a directory.
    """
    def __init__(self, root: StrPath, transform: Optional[Callable] = None, is_valid_file: Optional[Callable[[str], bool]] = None) -> None:
        """A Basic dataset that reads all images from a directory.

        Usage:
        ```python
            dataset = Basic("data/clic/train", T.Compose([
                                                   T.ToTensor(),
                                                   T.Normalize(mean, std)]))
        ```

        Args:
            root (str): The folder to read from.
            duplicate (int, optional): A number that repeat images in this dataset for N times. Defaults to 1.
            transform (Optional[Callable], optional): A transform applies to images. Defaults to None.
            is_valid_file (Optional[Callable[[str], bool]], optional): A function that check image is valid. Defaults to None, will use builtin implementation.

        Raises:
            RuntimeError: Find no images in this folder.
        """
        super().__init__(root, transform=transform)
        samples = _makeDataset(self.root, IMG_EXTENSIONS if is_valid_file is None else None, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            msg += "Supported extensions are: {}".format(",".join(IMG_EXTENSIONS))
            raise RuntimeError(msg)
        self.root = root
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples

    def __getitem__(self, index: int) -> Tuple[Tensor, str]:
        """
        Args:
            index (int): Index

        Returns:
            Tensor: sample at index.
        """
        path = self.samples[index]
        # sample = readImage(path)
        # sample = self.loader(path)
        sample = read_image(path, ImageReadMode.RGB)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, Path(path).stem

    def __len__(self) -> int:
        return len(self.samples)

    def __str__(self) -> str:
        return f"<Basic> at `{relativePath(self.root)}` with transform: \r\n`{self.transform}`"


class BasicLMDB(VisionDataset):
    """A Basic dataset that reads from a LMDB.
    """
    def __init__(self, root: StrPath, maxTxns: int = 1, repeat: int = 1, transform: Optional[Callable] = None) -> None:
        """A Basic dataset that reads from a LMDB.

        Usage:
        ```python
            dataset = Basic("data/trainSet", numWorkers + 2,
                               transform=T.Compose([T.ToTensor(),
                                                    T.Normalize(mean, std)]))
        ```

        Args:
            root (str): LMDB folder path. In addition, a `meatadata.json` should be placed in the same folder --- see `src/misc/datasetCreate.py`
            maxTxns (int, optional): Max trasactions of LMDB. Defaults to 1.
            repeat (int, optional): Repeat images of the dataset for N times. Defaults to 1.
            transform (Optional[Callable], optional): Transform applies to images. Defaults to None.
        """
        super().__init__(root, transform=transform)
        self._root = root
        self._maxTxns = maxTxns
        # env and txn is lazy-loaded in ddp. They can't be pickled
        self._env: Union[lmdb.Environment, None] = None
        self._txn: Union[lmdb.Transaction, None] = None
        # Length is needed for DistributedSampler, but we can't use env to get it, env can't be pickled.
        # So we decide to read from metadata placed in the same folder --- see src/misc/datasetCreate.py
        with open(os.path.join(root, "metadata.json"), "r") as fp:
            metadata = json.load(fp)
        self._length = metadata["length"]
        self._repeat = repeat

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._txn is not None:
            self._txn.__exit__(exc_type, exc_val, exc_tb)
        if self._env is not None:
            self._env.close()

    def _initEnv(self):
        self._env = lmdb.open(self.root, map_size=1024*1024*1024*8, subdir=True, readonly=True, readahead=False, meminit=False, max_spare_txns=self._maxTxns, lock=False)
        self._txn = self._env.begin(write=False, buffers=True)

    def __getitem__(self, index: int) -> Tensor:
        """
        Args:
            index (int): Index

        Returns:
            Tensor: sample at index.
        """
        index = index % self._length
        if self._env is None or self._txn is None:
            self._initEnv()
        sample = torch.ByteTensor(torch.ByteStorage.from_buffer(bytearray(self._txn.get(index.to_bytes(32, sys.byteorder))))) # type: ignore
        sample = decode_image(sample, ImageReadMode.RGB)
        if sample.shape[0] == 1:
            sample = sample.repeat((3, 1, 1))
        elif sample.shape[0] == 4:
            sample = sample[:3]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return self._length * self._repeat

    def __str__(self) -> str:
        return f"<LMDB> at `{relativePath(self.root)}` with transform: \r\n`{self.transform}`"
