from typing import Type

from vlutils.base import Registry

from mcquic.baseClass import ValueTuner

__all__ = [
    "ModuleRegistry",
    "ValueTunerRegistry",
    "LrSchedulerRegistry",
    "OptimizerRegistry",
    "LossRegistry",
    "HookRegistry"
]

class ModuleRegistry(Registry[Type["torch.nn.Module"]]):
    pass

class ValueTunerRegistry(Registry[Type[ValueTuner]]):
    pass

class LrSchedulerRegistry(Registry[Type["torch.optim.lr_scheduler._LRScheduler"]]):
    pass

class OptimizerRegistry(Registry[Type["torch.optim.optimizer.Optimizer"]]):
    pass

class LossRegistry(Registry[Type["torch.nn.Module"]]):
    pass

class HookRegistry(Registry[Type["mcquic.train.hooks.Hook"]]):
    pass
