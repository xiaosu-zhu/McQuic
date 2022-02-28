
import logging
import os

import mcquic

srcRoot = os.path.dirname(os.path.abspath(mcquic.__file__))

os.makedirs("/tmp/mcquic/", exist_ok=True)
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
