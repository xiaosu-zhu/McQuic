
import logging
import os

import mcqc

srcRoot = os.path.join(os.path.dirname(os.path.abspath(mcqc.__file__)), os.pardir)

class Consts:
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
