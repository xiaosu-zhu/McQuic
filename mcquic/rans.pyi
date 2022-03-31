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
    def decodeWithIndexes(self, encoded: str, indexes: typing.List[int], cdfs: typing.List[typing.List[int]], cdfSizes: typing.List[int], offsets: typing.List[int]) -> typing.List[int]: 
        """
        Decode a string to a list of symbols.

        This method is the reverse operation of `RansEncoder.encodeWithIndexes(...)` All args are same.

        Args:
            encoded (str): Encode byte string.
            indexes (List[int]): Index of CDF and cdfSize of i-th symbol to be used for encode.
            cdfs (List[List[int]]): A series of CDFs. Each corresponds to a group with specific PMF.
            cdfSizes (List[int]): Symbol upper-bound for each group.
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

        This method accepts symbols under mixed distributions. Therefore, symbol from different distribution can be encoded by its corresponding cdf to achieve the best rate.

        Args:
            symbols (List[int]): List of integers ranges in [0, cdfSize[index]] to be encoded.
            indexes (List[int]): Index of CDF and cdfSize of i-th symbol to be used for encode.
            cdfs (List[List[int]]): A series of CDFs. Each corresponds to a group with specific PMF.
            cdfSizes (List[int]): Symbol upper-bound for each group.
            offsets (List[int]): Offset applied to each symbol.

        Returns:
            str: Encoded byte string.
        """
    pass
def pmfToQuantizedCDF(arg0: typing.List[float], arg1: int) -> typing.List[int]:
    """
    Return quantized CDF for a given PMF with total quantization level `2 ** precision`.

    Args:
        pmf (List[float]): Probability mass function (normalized occuring frequency) for all symbols.
        precision (int): Total quantization level for CDF (cdfSize).

    return:
        List[int]: Quantized CDF, length = `2 ** precision`.
    """
