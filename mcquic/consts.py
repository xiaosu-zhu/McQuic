
import logging
import os
import shutil
import tempfile
import atexit


class ConstsMetaClass(type):
    @property
    def TempDir(cls):
        if getattr(cls, '_tempDir', None) is None:
            tempDir = os.path.dirname(tempfile.mktemp())
            tempDir = os.path.join(tempDir, "mcquic")
            cls._tempDir = tempDir
            os.makedirs(cls._tempDir, exist_ok=True)
            def removeTmp():
                shutil.rmtree(tempDir, ignore_errors=True)
            atexit.register(removeTmp)
        return cls._tempDir

class Consts(metaclass=ConstsMetaClass):
    Name = "mcquic"
    # lazy load
    # TempDir = "/tmp/mcquic/"
    Eps = 1e-6
    CDot = "·"
    TimeOut = 30
