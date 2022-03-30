from dataclasses import dataclass
from typing import List, Union

import msgpack
from marshmallow import Schema, fields, post_load, ValidationError
import vlutils.logger

from mcquic.utils import versionCheck


# TODO: goto marshmallow


class BytesField(fields.Field):
    def _validate(self, value):
        if not isinstance(value, bytes):
            raise ValidationError('Invalid input type.')
        if value is None or value == b'':
            raise ValidationError('Invalid value')


class ImageSizeSchema(Schema):
    height = fields.Int()
    width = fields.Int()
    channel = fields.Int()
    @post_load
    def _(self, data, **kwargs):
        return ImageSize(**data)

class CodeSizeSchema(Schema):
    m = fields.Int()
    heights = fields.List(fields.Int())
    widths = fields.List(fields.Int())
    k = fields.List(fields.Int())
    @post_load
    def _(self, data, **kwargs):
        return CodeSize(**data)

class FileHeaderSchema(Schema):
    qp = fields.Str()
    version = fields.Str()
    codeSize = fields.Nested(CodeSizeSchema())
    imageSize = fields.Nested(ImageSizeSchema())
    @post_load
    def _(self, data, **kwargs):
        return FileHeader(**data)


class FileSchema(Schema):
    fileHeader = fields.Nested(FileHeaderSchema())
    contents = fields.List(BytesField())
    @post_load
    def _(self, data, **kwargs):
        return File(**data)

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

    @property
    def Pixels(self) -> int:
        return self.height * self.width

    def __str__(self) -> str:
        return f"[{self.width}×{self.height}, {self.channel}]"


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

    def __str__(self) -> str:
        sequence = ", ".join(f"[{w}×{h}, {k}]" for h, w, k in zip(self.heights, self.widths, self.k))
        return f"""
        {self.m} code-groups: {sequence}"""


@dataclass(init=False)
class FileHeader:
    qp: str
    version: str
    codeSize: CodeSize
    imageSize: ImageSize
    def __init__(self, version: str, qp: str, codeSize: CodeSize, imageSize: ImageSize) -> None:
        if versionCheck(version):
            self.qp = qp
            self.version = version
            self.codeSize = codeSize
            self.imageSize = imageSize

    @property
    def QuantizationParameter(self) -> str:
        return str(self.qp)

    @property
    def Version(self) -> str:
        return self.version

    @property
    def CodeSize(self) -> CodeSize:
        return self.codeSize

    @property
    def ImageSize(self) -> ImageSize:
        return self.imageSize

    def __str__(self) -> str:
        return f"""
    Version    : {self.Version}
    QP         : {self.QuantizationParameter}
    Image size : {self.ImageSize}
    Code size  : {self.CodeSize}"""

@dataclass
class File:
    fileHeader: FileHeader
    contents: List[bytes]

    @property
    def FileHeader(self):
        return self.fileHeader

    @property
    def Content(self):
        return self.contents

    def serialize(self) -> bytes:
        thisFile: dict = FileSchema().dump(self)
        return msgpack.packb(thisFile, use_bin_type=True)

    @staticmethod
    def deserialize(data: bytes) -> "File":
        thisFile = msgpack.unpackb(data, use_list=False, raw=False)
        return FileSchema().load(thisFile)

    @property
    def BPP(self) -> float:
        return sum(len(x) for x in self.contents) * 8 / self.FileHeader.ImageSize.Pixels

    def size(self, human: bool = False) -> Union[int, str]:
        """Compute size of compressed binary, in bytes.

        Args:
            human (bool, optional): Whether to give a human-readable string (like `-h` option in POSIX). Defaults to False.

        Returns:
            Union[int, str]: If `human` is True, return integer of total bytes, else return human-readable string.
        """
        size = sum(len(x) for x in self.contents)
        if not human:
            return size
        return vlutils.logger.readableSize(size)

    def __str__(self) -> str:
        return f"""Header: {self.fileHeader}
Size  : {self.size(True)}
BPP   : {self.BPP:.4f}"""

    def __hash__(self) -> int:
        return hash(self.serialize())
