from typing import Optional, Any, Union, Sequence

import os
from torch import device, Tensor
import torch
from torch.nn import Module
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from mcquic.loss import Distortion, Rate
from mcquic.loss.lpips import LPIPS
from mcquic.modules.compressor import BaseCompressor

_device_t = Union[int, device]
_devices_t = Sequence[_device_t]


class Compound(Module):
    def __init__(self, compressor: BaseCompressor, distortion: Distortion, lpips: LPIPS):
        super().__init__()
        self._compressor = compressor
        self._distortion = distortion
        self._lpips = lpips

    def train(self, mode: bool = True):
        retValue = super().train(mode)
        self._lpips.eval()
        self._distortion.eval()
        return retValue

    def forward(self, x: Tensor):
        xHat, yHat, codes, logits = self._compressor(x)
        distortion = self._distortion(xHat, x, codes, logits)
        lpips = self._lpips(xHat, x)
        return xHat, (distortion, lpips.mean()), codes, logits

    @property
    def Freq(self):
        return self._compressor._quantizer._entropyCoder.NormalizedFreq

    @property
    def Compressor(self):
        return self._compressor

    def refresh(self, rank: int) -> torch.Tensor:
        if rank == 0:
            proportion = self.Compressor.reAssignCodebook()
        else:
            proportion = torch.zeros(())
        self.Compressor.syncCodebook()
        return proportion

    def formatDistortion(self, loss: torch.Tensor):
        return self._distortion.formatDistortion(loss)


# class Compound(DistributedDataParallel):
#     module: _compound
#     def __init__(self, compressor: BaseCompressor, criterion: Distortion, device_ids: Optional[_devices_t] = None, output_device: Optional[_device_t] = None, dim: int = 0, broadcast_buffers: bool = True, process_group: Optional[Any] = None, bucket_cap_mb: int = 25, find_unused_parameters: bool = False, **kwargs):
#         module = _compound(compressor, criterion)
#         super().__init__(module, device_ids, output_device, dim, broadcast_buffers, process_group, bucket_cap_mb, find_unused_parameters, **kwargs)
