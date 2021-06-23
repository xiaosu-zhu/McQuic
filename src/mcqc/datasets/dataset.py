from typing import Callable, Any, Optional, Tuple, Callable, List, Dict, cast
import os

import torch
from torchvision.io import read_image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from torchvision.io.image import ImageReadMode


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
    def __init__(self, root: str, transform: Optional[Callable] = None, is_valid_file: Optional[Callable[[str], bool]] = None) -> None:
        super().__init__(root, transform=transform)

        samples = make_dataset(self.root, IMG_EXTENSIONS if is_valid_file is None else None, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            msg += "Supported extensions are: {}".format(",".join(IMG_EXTENSIONS))
            raise RuntimeError(msg)

        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS

        self.samples = samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = read_image(path, ImageReadMode.RGB)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self.samples)
