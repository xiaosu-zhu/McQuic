import abc

from vlutils.base import Restorable


class Serializable(abc.ABC):
    @abc.abstractmethod
    def serialize(self):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def deserialize(raw):
        raise NotImplementedError


class ValueTuner(Restorable):
    def __init__(self, initValue: float = 2e-2):
        super().__init__()
        self._epoch = 0
        self._initValue = initValue

    def step(self):
        self._epoch += 1
        self.calc()

    def calc(self):
        self._value = self._initValue

    @property
    def Value(self) -> float:
        if not hasattr(self, "_value"):
            self.calc()
        return self._value
