from re import I
from typing import Optional, Any, Union, Sequence

from torch import device, Tensor
import torch
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel

from .compressor import BaseCompressor

_device_t = Union[int, device]
_devices_t = Sequence[_device_t]


class _composed(Module):
    def __init__(self, compressor: BaseCompressor, criterion: Module) -> None:
        super().__init__()
        self._compressor = compressor
        self._criterion = criterion

    def forward(self, x: Tensor):
        xHat, yHat, codes, logits = self._compressor(x)
        rate, distortion = self._criterion(x, xHat, codes, logits)
        return xHat, (rate, distortion), codes, logits

    def readyForCoding(self):
        return self._compressor.readyForCoding()

    @property
    def Freq(self):
        return self._compressor._quantizer._entropyCoder.Freq
class Composed(DistributedDataParallel):
    def __init__(self, compressor: BaseCompressor, criterion: Module, device_ids: Optional[_devices_t] = None, output_device: Optional[_device_t] = None, dim: int = 0, broadcast_buffers: bool = True, process_group: Optional[Any] = None, bucket_cap_mb: float = 25, find_unused_parameters: bool = False):
        module = _composed(compressor, criterion)
        self.module: _composed
        super().__init__(module, device_ids, output_device, dim, broadcast_buffers, process_group, bucket_cap_mb, find_unused_parameters)

    @property
    def Compressor(self):
        return self.module._compressor

    def refresh(self, rank: int) -> torch.Tensor:
        if rank == 0:
            proportion = self.Compressor.reAssignCodebook()
        else:
            proportion = torch.zeros(())
        self.Compressor.syncCodebook()
        return proportion
