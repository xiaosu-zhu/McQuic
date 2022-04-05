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
    # torch.finfo(torch.float).eps
    Eps = 1.1920928955078125e-07
    CDot = "·"
    TimeOut = 15
