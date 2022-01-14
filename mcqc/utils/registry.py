from typing import Type

from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from vlutils.base import Registry

from mcqc.base import ValueTuner

__all__ = [
    "ModuleRegistry",
    "ValueTunerRegistry",
    "LrSchedulerRegistry",
    "OptimizerRegistry"
]

class ModuleRegistry(Registry[Type[nn.Module]]):
    pass

class ValueTunerRegistry(Registry[Type[ValueTuner]]):
    pass

class LrSchedulerRegistry(Registry[Type[_LRScheduler]]):
    pass

class OptimizerRegistry(Registry[Type[Optimizer]]):
    pass
