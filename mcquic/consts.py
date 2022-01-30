
import logging
import os
import functools

import torch
import apex

import mcquic
from mcquic.utils import OptimizerRegistry, LrSchedulerRegistry

srcRoot = os.path.dirname(os.path.abspath(mcquic.__file__))

class Consts:
    Name = "mcquic"
    Fingerprint = "aHR0cHM6Ly9naXRodWIuY29tL3hpYW9zdS16aHUvbWNxYw=="
    CheckpointName = "saved.ckpt"
    DumpConfigName = "config.json"
    NewestDir = "latest"
    LoggerName = "main"
    RootDir = srcRoot
    LogDir = os.path.abspath(os.path.join(srcRoot, "../log"))
    TempDir = "/tmp/mcquic/"
    DataDir = os.path.abspath(os.path.join(srcRoot, "../data"))
    SaveDir = os.path.abspath(os.path.join(srcRoot, "../saved"))
    Logger = logging.getLogger(LoggerName)
    Eps = 1e-6
    CDot = "·"
    TimeOut = 15


OptimizerRegistry.register("Adam")(torch.optim.Adam)
OptimizerRegistry.register("Lamb")(functools.partial(apex.optimizers.FusedLAMB, set_grad_none=True))


LrSchedulerRegistry.register("ReduceLROnPlateau")(torch.optim.lr_scheduler.ReduceLROnPlateau)
LrSchedulerRegistry.register("Exponential")(torch.optim.lr_scheduler.ExponentialLR)
LrSchedulerRegistry.register("MultiStep")(torch.optim.lr_scheduler.MultiStepLR)
LrSchedulerRegistry.register("OneCycle")(torch.optim.lr_scheduler.OneCycleLR)
