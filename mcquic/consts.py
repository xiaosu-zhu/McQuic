
import logging
import os


class ConstsMetaClass(type):
    @property
    def TempDir(cls):
        if getattr(cls, '_tempDir', None) is None:
            os.makedirs("/tmp/mcquic/", exist_ok=True)
            cls._tempDir = "/tmp/mcquic/"
        return cls._tempDir

class Consts(metaclass=ConstsMetaClass):
    Name = "mcquic"
    Fingerprint = "aHR0cHM6Ly9naXRodWIuY29tL3hpYW9zdS16aHUvbWNxYw=="
    LoggerName = "root"
    # lazy load
    # TempDir = "/tmp/mcquic/"
    Logger = logging.getLogger(LoggerName)
    Eps = 1e-6
    CDot = "·"
    TimeOut = 15
