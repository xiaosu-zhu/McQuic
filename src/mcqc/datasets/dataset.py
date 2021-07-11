from typing import Callable, Any, Optional, Tuple, Callable, List, cast
import os
import json

import lmdb
import torch
from torchvision.io import read_image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from torchvision.io.image import ImageReadMode, decode_image


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def make_dataset(directory: str, extensions: Optional[Tuple[str, ...]] = None, is_valid_file: Optional[Callable[[str], bool]] = None,) -> List[str]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    def validFileWrapper(x):
        return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    if extensions is not None:
        is_valid_file = validFileWrapper
    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                instances.append(path)
    return instances


class Basic(VisionDataset):
    def __init__(self, root: str, duplicate: int = 1, transform: Optional[Callable] = None, is_valid_file: Optional[Callable[[str], bool]] = None) -> None:
        super().__init__(root, transform=transform)
        samples = make_dataset(self.root, IMG_EXTENSIONS if is_valid_file is None else None, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            msg += "Supported extensions are: {}".format(",".join(IMG_EXTENSIONS))
            raise RuntimeError(msg)
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples * duplicate

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        # sample = readImage(path)
        # sample = self.loader(path)
        sample = read_image(path, ImageReadMode.RGB)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.samples)


class BasicLMDB(VisionDataset):
    def __init__(self, root: str, maxTxns: int = 1, transform: Optional[Callable] = None, is_valid_file: Optional[Callable[[str], bool]] = None) -> None:
        super().__init__(root, transform=transform)
        self._maxTxns = maxTxns
        # env and txn is delay-loaded in ddp. They can't pickle
        self._env = None
        self._txn = None
        # Length is needed for DistributedSampler, but we can't use env to get it, env can't pickle.
        # So we decide to read from metadata placed in the same folder --- see src/misc/datasetCreate.py
        with open(os.path.join(root, "metadata.json"), "r") as fp:
            metadata = json.load(fp)
        self._length = metadata["length"]

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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self._env is None:
            self._initEnv()
        sample = torch.ByteTensor(torch.ByteStorage.from_buffer(bytearray(self._txn.get(index.to_bytes(32, "big")))))
        sample = decode_image(sample, ImageReadMode.UNCHANGED)
        if sample.shape[0] == 1:
            sample = sample.repeat((3, 1, 1))
        elif sample.shape[0] == 4:
            sample = sample[:3]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return self._length
