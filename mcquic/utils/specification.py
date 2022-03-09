import abc
from dataclasses import dataclass
from typing import List, Union

import vlutils.logger

import mcquic


# TODO: goto marshmallow

class Serializable(abc.ABC):
    @abc.abstractmethod
    def serialize(self):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def deserialize(raw):
        raise NotImplementedError


SEPARATOR = "@|@"


@dataclass
class ImageSize(Serializable):
    """Image size specification.

    Args:
        height (int): Image height.
        width (int): Image width.
        channel (int): Image channel.
    """
    height: int
    width: int
    channel: int

    @property
    def Pixels(self) -> int:
        return self.height * self.width

    def serialize(self):
        return f"{self.height},{self.width},{self.channel}"

    @staticmethod
    def deserialize(raw):
        height, width, channel = raw.split(",")
        return ImageSize(int(height), int(width), int(channel))

    def __str__(self) -> str:
        return f"[{self.width}×{self.height}, {self.channel}]"


@dataclass
class CodeSize(Serializable):
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

    def serialize(self):
        sequence = SEPARATOR.join(",".join(map(str, x)) for x in zip(self.heights, self.widths, self.k))
        return f"{self.m}+{sequence}"

    @staticmethod
    def deserialize(raw):
        m, sequence = raw.split("+")
        sequence = sequence.split(SEPARATOR)
        heights, widths, k = list(map(list, zip(*[s.split(",") for s in sequence])))
        return CodeSize(int(m), list(map(int, heights)), list(map(int, widths)), list(map(int, k)))

    def __str__(self) -> str:
        sequence = ", ".join(f"[{w}×{h}, {k}]" for h, w, k in zip(self.heights, self.widths, self.k))
        return f"""
        {self.m} code-groups: {sequence}"""


class FileHeader(Serializable):
    _sep = ":|:"
    def __init__(self, version: str, qp: str, codeSize: CodeSize, imageSize: ImageSize, strict: bool = True) -> None:
        if strict and mcquic.__version__ != version:
            raise ValueError("Version mismatch.")
        self._qp = qp
        self._version = version
        self._codeSize = codeSize
        self._imageSize = imageSize

    @property
    def QuantizationParameter(self) -> str:
        return str(self._qp)

    @property
    def Version(self) -> str:
        return self._version

    @property
    def CodeSize(self) -> CodeSize:
        return self._codeSize

    @property
    def ImageSize(self) -> ImageSize:
        return self._imageSize

    def serialize(self) -> str:
        return self._sep.join([self._version, self._qp, self._codeSize.serialize(), self._imageSize.serialize()])

    @staticmethod
    def deserialize(raw: str) -> "FileHeader":
        version, qp, codeSize, imageSize = raw.split(FileHeader._sep)
        return FileHeader(version, qp, CodeSize.deserialize(codeSize), ImageSize.deserialize(imageSize))

    def __str__(self) -> str:
        return f"""
    Version    : {self.Version}
    QP         : {self.QuantizationParameter}
    Image size : {self.ImageSize}
    Code size  : {self.CodeSize}"""


class File(Serializable):
    _bsep = b"/|/"
    _gsep = b"&|&"

    def __init__(self, header: FileHeader, content: List[bytes]):
        self._header = header
        self._content = content

    @property
    def Header(self):
        return self._header

    @property
    def Content(self):
        return self._content

    def serialize(self) -> bytes:
        contents = self._bsep.join(self._content)
        return self._gsep.join([self._header.serialize().encode("utf-8"), contents])

    @staticmethod
    def deserialize(raw: bytes) -> "File":
        headerBin, contents = raw.split(File._gsep)
        header = FileHeader.deserialize(headerBin.decode("utf-8"))
        content = contents.split(File._bsep)
        return File(header, content)

    @property
    def BPP(self) -> float:
        return sum(len(x) for x in self._content) * 8 / self.Header.ImageSize.Pixels

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
        return f"""Header: {self._header}
Size  : {self.size(True)}
BPP   : {self.BPP}"""
