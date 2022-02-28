import math

from mcquic.utils import ValueTunerRegistry
from mcquic.baseClass import ValueTuner

@ValueTunerRegistry.register
class CyclicValue(ValueTuner):
    def __init__(self, initValue: float = 2e-2, gamma: float = 1.0, cyclicInterval: int = 400, boostInterval: int = 3, zeroOutRatio: float = 1./3.):
        super().__init__(initValue=initValue)
        self._cyclicInterval = cyclicInterval
        self._boostInterval = boostInterval
        self._zeroOutRatio = zeroOutRatio
        self._gamma = gamma

    def calc(self):
        maxReg = self._initValue * (self._gamma ** self._epoch)
        # phase 1
        if (self._epoch // self._cyclicInterval) % self._boostInterval == 0:
            self._value = maxReg
        # phase 2
        else:
            j = (self._epoch % self._cyclicInterval) / float(self._cyclicInterval)
            down = 2 * maxReg / (self._zeroOutRatio - 1) * j + maxReg
            up = 2 * maxReg / (1 - self._zeroOutRatio) * j + (self._zeroOutRatio + 1) / (self._zeroOutRatio - 1) * maxReg
            self._value = max(0, max(up, down))


@ValueTunerRegistry.register
class ExponentialValue(ValueTuner):
    def __init__(self, initValue: float = 2e-2, gamma: float = 0.9999):
        super().__init__(initValue=initValue)
        self._gamma = gamma

    def calc(self):
        self._value = self._initValue * (self._gamma ** self._epoch)


@ValueTunerRegistry.register
class StepValue(ValueTuner):
    def __init__(self, initValue: float = 2e-2, gamma: float = 0.1, stepInterval: int = 1000):
        super().__init__(initValue=initValue)
        self._gamma = gamma
        self._stepInterval = stepInterval

    def calc(self):
        self._value = self._initValue * (self._gamma ** (self._epoch // self._stepInterval))

@ValueTunerRegistry.register
class CosineValue(ValueTuner):
    def __init__(self, maxValue: float = 1.0, minValue: float = 0.0, stepInterval: int = 1, totalStep: int = 1000, revert: bool = False):
        super().__init__(initValue=maxValue)
        self._minValue = minValue
        self._stepInterval = stepInterval
        self._totalStep = totalStep
        self._revert = revert

    def calc(self):
        # 1 ~ -1
        nowCosine = math.cos(math.pi * self._epoch / self._stepInterval / self._totalStep)
        if self._revert:
            nowCosine = nowCosine
        realValue = (nowCosine + 1) / 2.0 * (self._initValue - self._minValue) + self._minValue
        self._value = realValue

@ValueTunerRegistry.register
class CosineValueWithEnd(ValueTuner):
    def __init__(self, maxValue: float = 1.0, minValue: float = 0.0, stepInterval: int = 1, totalStep: int = 1000, revert: bool = False):
        super().__init__(initValue=maxValue)
        self._minValue = minValue
        self._stepInterval = stepInterval
        self._totalStep = totalStep
        self._revert = revert

    def calc(self):
        if self._epoch / self._stepInterval / self._totalStep >= 1:
            self._value = self._minValue
            return
        # 1 ~ -1
        nowCosine = math.cos(math.pi * self._epoch / self._stepInterval / self._totalStep)
        if self._revert:
            nowCosine = -nowCosine
        realValule = (nowCosine + 1) / 2.0 * (self._initValue - self._minValue) + self._minValue
        self._value = realValule


@ValueTunerRegistry.register
class JumpValue(ValueTuner):
    def __init__(self, initValue: float = 10.0, gamma: float = 0.9, stepInterval: int = 1000, minValue: float = 0.01):
        super().__init__(initValue=initValue)
        self._gamma = gamma
        self._stepInterval = stepInterval
        self._max = initValue
        self._min = minValue

        self._iteration = int(math.log(self._min / self._max) / math.log(self._gamma))

    def calc(self):
        self._value = self._initValue * (self._gamma ** ((self._epoch // self._stepInterval) % self._iteration))


@ValueTunerRegistry.register
class JumpAlter(ValueTuner):
    def __init__(self, initValue: float = 10.0, gamma: float = 0.9, stepInterval: int = 10, minValue: float = 0.01, milestone: int = 500, valueAfterMilestone: float = 0.01):
        super().__init__(initValue=initValue)
        self._gamma = gamma
        self._stepInterval = stepInterval
        self._max = initValue
        self._min = minValue
        self._milestone = milestone
        self._valueAfterMilestone = valueAfterMilestone

        self._iteration = int(math.log(self._min / self._max) / math.log(self._gamma))

    def calc(self):
        if self._iteration <= self._milestone:
            self._value = self._initValue * (self._gamma ** ((self._epoch // self._stepInterval) % self._iteration))
        else:
            self._value = self._valueAfterMilestone
