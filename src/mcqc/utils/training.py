from logging import warning
from typing import Any
import warnings
import math
from cfmUtils.base.restorable import Restorable
from collections import Counter
from bisect import bisect_right

import torch
from torch.optim.optimizer import Optimizer


class MovingMean:
    def __init__(self, momentum=0.9):
        super().__init__()
        self._slots = dict()
        self._alpha = 1 - momentum

    def step(self, key, value: torch.Tensor):
        if key in self._slots:
            mean = self._slots[key]
            mean -= self._alpha * (mean - float(value))
            self._slots[key] = mean
        else:
            mean = value.item()
            self._slots[key] = mean
        return mean

    def __getitem__(self, key: Any) -> float:
        return self._slots[key]



class MultiStepLRWithWarmUp(torch.optim.lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        firstMilestone = min(self.milestones)

        if self.last_epoch <= firstMilestone:
            return [base_lr * float(self.last_epoch + 1) / firstMilestone for base_lr in self.base_lrs]

        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]


class CyclicLR(torch.optim.lr_scheduler._LRScheduler):
    r"""Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This class has three built-in policies, as put forth in the paper:

    * "triangular": A basic triangular cycle without amplitude scaling.
    * "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
    * "exp_range": A cycle that scales initial amplitude by :math:`\text{gamma}^{\text{cycle iterations}}`
      at each cycle iteration.

    This implementation was adapted from the github repo: `bckenstler/CLR`_

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up. Default: None
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            If specified, then 'mode' is ignored.
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.8
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the difference of max_momentum
            and some scaling of the amplitude; therefore
            base_momentum may not actually be reached depending on
            scaling function. Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.9
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()


    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self,
                 optimizer,
                 base_lr,
                 max_lr,
                 step_size_up=2000,
                 step_size_down=None,
                 mode='triangular',
                 gamma=1.,
                 scale_fn=None,
                 scale_mode='cycle',
                 cycle_momentum=True,
                 base_momentum=0.8,
                 max_momentum=0.9,
                 last_epoch=-1,
                 verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        base_lrs = self._format_param('base_lr', optimizer, base_lr)
        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = lr

        self.max_lrs = self._format_param('max_lr', optimizer, max_lr)

        step_size_up = float(step_size_up)
        step_size_down = float(step_size_down) if step_size_down is not None else step_size_up
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.cycle_momentum = cycle_momentum
        if self.cycle_momentum:
            if 'momentum' not in self.optimizer.defaults and 'betas' not in self.optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')
            self.use_beta1 = 'betas' in self.optimizer.defaults
            max_momentums = self._format_param('max_momentum', optimizer, max_momentum)
            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for m_momentum, b_momentum, group in zip(max_momentums, base_momentums, optimizer.param_groups):
                    if self.use_beta1:
                        _, beta2 = group['betas']
                        group['betas'] = (m_momentum, beta2)
                    else:
                        group['momentum'] = m_momentum
                    group['max_momentum'] = m_momentum
                    group['base_momentum'] = b_momentum
            # self.base_momentums = base_momentums
            # self.max_momentums = max_momentums
        super(CyclicLR, self).__init__(optimizer, last_epoch, verbose)
        self.base_lrs = base_lrs

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """

        if not self._get_lr_called_within_step:
            warning.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1. + self.last_epoch / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            lrs.append(lr)

        if self.cycle_momentum:
            for group in self.optimizer.param_groups:
                base_momentum, max_momentum = group["base_momentum"], group["max_momentum"]
                base_height = (max_momentum - base_momentum) * scale_factor
                if self.scale_mode == 'cycle':
                    momentum = max_momentum - base_height * self.scale_fn(cycle)
                else:
                    momentum = max_momentum - base_height * self.scale_fn(self.last_epoch)
                if self.use_beta1:
                    _, beta2 = group['betas']
                    group['betas'] = (momentum, beta2)
                else:
                    group['momentum'] = momentum
        return lrs


class _ValueTuner(Restorable):
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


class CyclicValue(_ValueTuner):
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


class ExponentialValue(_ValueTuner):
    def __init__(self, initValue: float = 2e-2, gamma: float = 0.9999):
        super().__init__(initValue=initValue)
        self._gamma = gamma

    def calc(self):
        self._value = self._initValue * (self._gamma ** self._epoch)


class StepValue(_ValueTuner):
    def __init__(self, initValue: float = 2e-2, gamma: float = 0.1, stepInterval: int = 1000):
        super().__init__(initValue=initValue)
        self._gamma = gamma
        self._stepInterval = stepInterval

    def calc(self):
        self._value = self._initValue * (self._gamma ** (self._epoch // self._stepInterval))


class JumpValue(_ValueTuner):
    def __init__(self, initValue: float = 10.0, gamma: float = 0.9, stepInterval: int = 1000, minValue: float = 0.01):
        super().__init__(initValue=initValue)
        self._gamma = gamma
        self._stepInterval = stepInterval
        self._max = initValue
        self._min = minValue

        self._iteration = int(math.log(self._min / self._max) / math.log(self._gamma))

    def calc(self):
        self._value = self._initValue * (self._gamma ** ((self._epoch // self._stepInterval) % self._iteration))


class JumpAlter(_ValueTuner):
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


if __name__ == "__main__":
    a = JumpValue(10.0, 0.9931160484209338, 10, 0.01)
    from matplotlib import pyplot as plt
    values = list()
    for i in range(10000):
        a.step()
        values.append(float(a._value))
    plt.plot(values)
    plt.savefig("value.pdf")
