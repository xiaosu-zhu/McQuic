"""range Asymmetric Numeral System (rANS) python bindings.

Exports:
    RansEncoder: Encode list of symbols to string.
    RansDecoder: Decode a string to a list of symbols.
    pmfToQuantizedCDF: Return quantized CDF for a given PMF."""
from __future__ import annotations
import mcquic.rans
import typing

__all__ = [
    "RansDecoder",
    "RansEncoder",
    "pmfToQuantizedCDF"
]


class RansDecoder():
    """
    Decoder to decode a string to a list of symbols. This class exports only one method `decodeWithIndexes(...)`.
    """
    def __init__(self) -> None: ...
    def decodeWithIndexes(self, encoded: bytes, indexes: typing.List[int], cdfs: typing.List[typing.List[int]], cdfSizes: typing.List[int], offsets: typing.List[int]) -> typing.List[int]: 
        """
        Decode a string to a list of symbols.

        This method is inverse operation of `RansEncoder.encodeWithIndexes(...)`. All args are same.

        Args:
            encoded (bytes): Encoded byte string.
            indexes (List[int]): Index of CDF and cdfSize to pick for i-th symbol.
            cdfs (List[List[int]]): A series of CDFs. Each corresponds to a different symbol group. Different groups have different CDFs since they are under different distributions.
            cdfSizes (List[int]): Symbol integer upper-bound for each group.
            offsets (List[int]): Offset applied to each symbol.

        Returns:
            List[int]: Decoded symbol list.
        """
    pass
class RansEncoder():
    """
    Encoder to encode list of symbols to string. This class exports only one method `encodeWithIndexes(...)`.
    """
    def __init__(self) -> None: ...
    def encodeWithIndexes(self, symbols: typing.List[int], indexes: typing.List[int], cdfs: typing.List[typing.List[int]], cdfSizes: typing.List[int], offsets: typing.List[int]) -> bytes: 
        """
        Encode list of symbols to string.

        This method accepts symbols under mixed distributions. Therefore, symbol from different distribution can be encoded by its corresponding CDF to achieve the best rate.

        Args:
            encoded (bytes): Encoded byte string.
            indexes (List[int]): Index of CDF and cdfSize to pick for i-th symbol.
            cdfs (List[List[int]]): A series of CDFs. Each corresponds to a different symbol group. Different groups have different CDFs since they are under different distributions.
            cdfSizes (List[int]): Symbol integer upper-bound for each group.
            offsets (List[int]): Offset applied to each symbol.

        Returns:
            bytes: Encoded byte string.
        """
    pass
def pmfToQuantizedCDF(pmf: typing.List[float], precision: int) -> typing.List[int]:
    """
    Return quantized CDF for a given PMF with total quantization level `2 ** precision`.

    Args:
        pmf (List[float]): Probability mass function (normalized symbol frequency).
        precision (int): Total quantization level in bits (`log2(cdfSize)`).

    return:
        List[int]: Quantized CDF, length = `2 ** precision`.
    """
