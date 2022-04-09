from typing import Optional, Any, Union, Sequence

from torch import device, Tensor
import torch
from torch.nn import Module
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from mcquic.loss import Distortion
from mcquic.datasets.transforms import getTraingingPostprocess
from .compressor import BaseCompressor

_device_t = Union[int, device]
_devices_t = Sequence[_device_t]


class _compound(Module):
    def __init__(self, compressor: BaseCompressor, criterion: Distortion):
        super().__init__()
        self._compressor = compressor
        self._criterion = criterion
        self._postProcess = getTraingingPostprocess().to(dist.get_rank())
        self._postProcessEnabled = True

    def forward(self, x: Tensor):
        if self._postProcessEnabled:
            post = self._postProcess(x)
        else:
            post = x
        xHat, yHat, codes, logits = self._compressor(post)
        rate, distortion = self._criterion(x, xHat, codes, logits)
        return (post, xHat), (rate, distortion), codes, logits

    @property
    def Freq(self):
        return self._compressor._quantizer._entropyCoder.NormalizedFreq

    @property
    def PostProcessEnabled(self):
        return self._postProcessEnabled

    @PostProcessEnabled.setter
    def PostProcessEnabled(self, enabled: bool):
        self._postProcessEnabled = enabled


class Compound(DistributedDataParallel):
    module: _compound
    def __init__(self, compressor: BaseCompressor, criterion: Distortion, device_ids: Optional[_devices_t] = None, output_device: Optional[_device_t] = None, dim: int = 0, broadcast_buffers: bool = True, process_group: Optional[Any] = None, bucket_cap_mb: float = 25, find_unused_parameters: bool = False, **kwargs):
        module = _compound(compressor, criterion)
        super().__init__(module, device_ids, output_device, dim, broadcast_buffers, process_group, bucket_cap_mb, find_unused_parameters, **kwargs)

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

    def formatDistortion(self, loss: torch.Tensor):
        return self.module._criterion.formatDistortion(loss)

    @property
    def PostProcessEnabled(self):
        return self.module.PostProcessEnabled

    @PostProcessEnabled.setter
    def PostProcessEnabled(self, enabled: bool):
        self.module.PostProcessEnabled = enabled
