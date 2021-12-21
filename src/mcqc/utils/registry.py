from vlutils.base import Registry


__all__ = [
    "ModuleRegistry",
    "ValueTunerRegistry",
    "LrSchedulerRegistry",
    "OptimizerRegistry"
]

class ModuleRegistry(Registry):
    pass

class ValueTunerRegistry(Registry):
    pass

class LrSchedulerRegistry(Registry):
    pass

class OptimizerRegistry(Registry):
    pass
