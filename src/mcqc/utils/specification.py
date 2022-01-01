
from dataclasses import dataclass
from typing import List


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
