
from dataclasses import dataclass
import os
from typing import List, Union

import vlutils.logger

@dataclass
class ImageSize:
    """Image size specification.

    Args:
        height (int): Image height.
        width (int): Image width.
        channel (int): Image channel.
    """
    height: int
    width: int
    channel: int


@dataclass
class CodeSize:
    """Latent code specification.
           Code in this paper is of shape: `[[1, m, h, w], [1, m, h, w] ... ]`
                                                            `↑ total length = L`

    Args:
        heights (List[int]): Latent height for each stage.
        widths (List[int]): Latent width for each stage.
        k (List[int]): [k1, k2, ...], codewords amount for each stage.
        m (int): M, multi-codebook amount.
    """
    m: int
    heights: List[int]
    widths: List[int]
    k: List[int]


class FileHeader:
    _sep = ":|:"
    def __init__(self, fingerprint: str, codeSize: CodeSize, imageSize: ImageSize) -> None:
        self._fingerprint = fingerprint
        self._codeSize = codeSize
        self._imageSize = imageSize


    @property
    def Fingerprint(self) -> str:
        return self._fingerprint

    @property
    def CodeSize(self) -> CodeSize:
        return self._codeSize

    @property
    def ImageSize(self) -> ImageSize:
        return self._imageSize

    def serialize(self) -> str:
        return self._sep.join([self._fingerprint, str(self._codeSize), str(self._imageSize)])

    @staticmethod
    def deserialize(raw: str) -> "FileHeader":
        fingerprint, codeSize, imageSize = raw.split(FileHeader._sep)
        return FileHeader(fingerprint, eval(codeSize), eval(imageSize))


class File:
    def __init__(self, header: FileHeader, content: List[bytes]):
        self._header = header
        self._content = content

    def size(self, human: bool = False) -> Union[int, str]:
        """Compute size of compressed binary, in bytes.

        Args:
            human (bool, optional): Whether to give a human-readable string (like `-h` option in POSIX). Defaults to False.

        Returns:
            Union[int, str]: If `human` is True, return integer of total bytes, else return human-readable string.
        """
        size = sum(len(x) for x in self._content)
        if not human:
            return size
        return vlutils.logger.readableSize(size)

    def __str__(self) -> str:
        return f"""
            Header: {self._header}{os.sep}
            Size  : {self.size(True)}
        """
