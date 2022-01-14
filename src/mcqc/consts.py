
import logging
import os
import functools

import torch
import apex

import mcqc
from mcqc.utils import OptimizerRegistry, LrSchedulerRegistry

srcRoot = os.path.join(os.path.dirname(os.path.abspath(mcqc.__file__)), os.pardir)

class Consts:
    Name = "mcqc"
    Fingerprint = "aHR0cHM6Ly9naXRodWIuY29tL3hpYW9zdS16aHUvbWNxYw=="
    CheckpointName = "saved.ckpt"
    DumpConfigName = "config.json"
    NewestDir = "latest"
    LoggerName = "main"
    RootDir = srcRoot
    LogDir = os.path.join(srcRoot, "../log")
    TempDir = "/tmp/mcqc/"
    DataDir = os.path.join(srcRoot, "../data")
    SaveDir = os.path.join(srcRoot, "../saved")
    Logger = logging.getLogger(LoggerName)
    Eps = 1e-6
    CDot = "·"



OptimizerRegistry.register("Adam")(torch.optim.Adam)
OptimizerRegistry.register("Lamb")(functools.partial(apex.optimizers.FusedLAMB, set_grad_none=True))


LrSchedulerRegistry.register("ReduceLROnPlateau")(torch.optim.lr_scheduler.ReduceLROnPlateau)
LrSchedulerRegistry.register("Exponential")(torch.optim.lr_scheduler.ExponentialLR)
LrSchedulerRegistry.register("MultiStep")(torch.optim.lr_scheduler.MultiStepLR)
LrSchedulerRegistry.register("OneCycle")(torch.optim.lr_scheduler.OneCycleLR)
