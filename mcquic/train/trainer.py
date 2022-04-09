import functools
import os
import shutil
from time import sleep
from typing import Callable, Optional, Tuple, Type
import signal
import threading
import gc

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch import distributed as dist
from vlutils.base.freqHook import ChainHook
from vlutils.saver import Saver
from vlutils.logger import trackingFunctionCalls
from vlutils.base import Restorable
from vlutils.runtime import relativePath
from rich import filesize

from mcquic.consts import Consts
from mcquic.loss import Distortion
from mcquic.validate.utils import EMATracker
from mcquic.modules.compound import Compound
from mcquic import Config
from mcquic.modules.compressor import BaseCompressor
from mcquic.validate import Validator
from mcquic.utils import totalParameters
from mcquic.datasets.transforms import getTrainingTransform

from .utils import EpochFrequencyHook, checkHook, getRichProgress

class _baseTrainer(Restorable):
    def __init__(self, config: Config, modelFn: Callable[[], Tuple[BaseCompressor, Distortion]], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], saver: Saver, **_):
        super().__init__()
        self.saver = saver

        self.rank = dist.get_rank()
        self.saver.debug("<%s> is located at rank `%d`", self.__class__.__name__, self.rank)
        self.worldSize = dist.get_world_size()
        torch.cuda.set_device(self.rank)
        self.config = config

        self._epoch = 0
        self._step = 0

        # Used for self.PrettyStep
        self.lastFormatted = -1
        self.prettyStep = "......"

        self.transform = getTrainingTransform().to(self.rank)

        self.saver.debug("[%s] Creating model...", self.PrettyStep)
        compressor, criterion = trackingFunctionCalls(modelFn, self.saver)()
        self._model = Compound(compressor.to(self.rank), criterion.to(self.rank), device_ids=[self.rank], output_device=self.rank)
        self.saver.debug("[%s] Model created.", self.PrettyStep)
        self.saver.debug("[%s] Model size: %s", self.PrettyStep, totalParameters(self._model))

        self.saver.debug("[%s] Creating optimizer...", self.PrettyStep)
        optimizer = trackingFunctionCalls(optimizer, self.saver)
        self._optimizer = optimizer(self._model.parameters(), **self.config.Train.Optim.Params)
        self.optimFn = optimizer
        self.saver.debug("[%s] Optimizer created.", self.PrettyStep)

        self.saver.debug("[%s] Creating LR scheduler...", self.PrettyStep)
        scheduler = trackingFunctionCalls(scheduler, self.saver)
        self._scheduler = scheduler(self._optimizer, **self.config.Train.Schdr.Params)
        self.schdrFn = scheduler
        self.saver.debug("[%s] LR scheduler created.", self.PrettyStep)

        self.saver.debug("[%s] <%s> created.", self.PrettyStep, self.__class__.__name__)

    def save(self, path = None):
        self.saver.save(path, trainer=self, config=self.config.serialize())

    @property
    def PrettyStep(self):
        if self._step == self.lastFormatted:
            return self.prettyStep
        else:
            self.prettyStep = self._formatStep(self._step)
            self.lastFormatted = self._step
            return self.prettyStep

    @staticmethod
    def _formatStep(step):
        unit, suffix = filesize.pick_unit_and_suffix(step, ["", "k", "M"], 1000)
        if unit < 10:
            return f"{(step // unit):5d}"
        else:
            truncated = step / unit
            if truncated < 10:
                return f"{truncated:4.6f}"[:4] + suffix
            elif truncated < 100:
                return f"{truncated:4.6f}"[:4] + suffix
            else:
                return f"{truncated:11.6f}"[:4] + suffix

    def restoreStates(self, path: str):
        self.saver.debug("[%s] Restored state dict from `%s`", self.PrettyStep, path)

        self.saver.load(path, torch.device(f"cuda:{self.rank}"), logger=self.saver, trainer=self)

        self.saver.debug("[%s] Restore network parameters finished.", self.PrettyStep)

        self.resetOptimizer()
        for group in self._optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.resetScheduler(self._scheduler.last_epoch)

    def resetOptimizer(self):
        del self._optimizer
        self._optimizer = self.optimFn(self._model.parameters(), **self.config.Train.Optim.Params)

        self.saver.debug("[%s] Optimizer reset.", self.PrettyStep)

    def resetScheduler(self, lastEpoch=-1):
        del self._scheduler
        self._scheduler = self.schdrFn(self._optimizer, last_epoch=lastEpoch, **self.config.Train.Schdr.Params)

        self.saver.debug("[%s] LR scheduler reset.", self.PrettyStep)

    def _beforeRun(self, hook, *args, **kwArgs):
        if self._step > 0:
            self.saver.info("[%s] Resume training at %s steps / %d epochs.", self.PrettyStep, self.PrettyStep, self._epoch)
        else:
            self.saver.info("[%s] Start training.", self.PrettyStep)

        self.saver.debug("[%s] Training loop started.", self.PrettyStep)
        hook(self._step, self._epoch, self, *args, logger=self.saver, **kwArgs)

    def _afterRun(self, hook, *args, **kwArgs):
        self.saver.debug("[%s] Training loop finished.", self.PrettyStep)
        hook(self._step, self._epoch, self, *args, logger=self.saver, **kwArgs)

    def _stepStart(self, hook, *args, **kwArgs):
        hook(self._step, self._epoch, self, *args, logger=self.saver, **kwArgs)

    def _stepFinish(self, hook, *args, **kwArgs):
        self._step += 1
        hook(self._step, self._epoch, self, *args, logger=self.saver, **kwArgs)

    def _epochStart(self, hook, *args, trainSampler, **kwArgs):
        trainSampler.set_epoch(self._epoch)

        self.saver.debug("[%s] Epoch %4d started.", self.PrettyStep, self._epoch + 1)

        gc.collect()
        gc.collect()
        hook(self._step, self._epoch, self, *args, trainSampler=trainSampler, logger=self.saver, **kwArgs)

    def _epochFinish(self, hook, *args, **kwArgs):
        self._epoch += 1

        self.saver.debug("[%s] Epoch %4d finished.", self.PrettyStep, self._epoch)

        self._scheduler.step()
        self.saver.debug("[%s] Lr is set to %.2e.", self.PrettyStep, self._scheduler.get_last_lr()[0])

        hook(self._step, self._epoch, self, *args, logger=self.saver, **kwArgs)


    def train(self, trainLoader: DataLoader, trainSampler: DistributedSampler, *_, beforeRunHook: Optional[Callable] = None, afterRunHook: Optional[Callable] = None, epochStartHook: Optional[Callable] = None, epochFinishHook: Optional[Callable] = None, stepStartHook: Optional[Callable] = None, stepFinishHook: Optional[Callable] = None, **__):
        beforeRunHook = checkHook(beforeRunHook, "BeforeRunHook", self.saver)
        afterRunHook = checkHook(afterRunHook, "AfterRunHook", self.saver)
        stepStartHook = checkHook(stepStartHook, "StepStartHook", self.saver)
        stepFinishHook = checkHook(stepFinishHook, "StepFinishHook", self.saver)
        epochStartHook = checkHook(epochStartHook, "EpochStartHook", self.saver)
        epochFinishHook = checkHook(epochFinishHook, "EpochFinishHook", self.saver)

        trainingArgs = {
            "trainLoader": trainLoader,
            "trainSampler": trainSampler
        }

        self._beforeRun(beforeRunHook, **trainingArgs)

        images, postProcessed, xHat, codes, logits = None, None, None, None, None

        for _ in range(self._epoch, self.config.Train.Epoch):
            self._epochStart(epochStartHook, **trainingArgs)
            for images in trainLoader:
                images = self.transform(images.to(self.rank, non_blocking=True))

                self._stepStart(stepStartHook, **trainingArgs)

                self._optimizer.zero_grad()

                (postProcessed, xHat), (rate, distortion), codes, logits = self._model(images)
                (rate + distortion).backward()

                self._optimizer.step()

                self._stepFinish(stepFinishHook, rate=rate, distortion=distortion, **trainingArgs)
            self._epochFinish(epochFinishHook, images=images, restored=xHat, postProcessed=postProcessed, codes=codes, logits=logits, trainSet=trainLoader.dataset, **trainingArgs)
        self._afterRun(afterRunHook)


class MainTrainer(_baseTrainer):
    def __init__(self, config: Config, modelFn: Callable[[], Tuple[BaseCompressor, Distortion]], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], saver: Saver):
        if dist.get_rank() != 0:
            raise AttributeError("A sub-process should not to be a <MainTrainer>, use <PalTrainer> instead.")

        self.rank = dist.get_rank()
        self.config = config
        self.saver = saver

        self.progress = getRichProgress().__enter__()
        self.trainingBar = self.progress.add_task("", start=False, progress="[----/----]", suffix=Consts.CDot * 10)
        self.epochBar = self.progress.add_task("[----/----]", start=False, progress="", suffix=Consts.CDot * 10)

        self.validator = Validator(self.config, self.rank)

        self.diffTracker = EMATracker((), 0.99).to(self.rank)

        # Call function at every X epoches.
        self.epochFinishCalls = EpochFrequencyHook(
            (1, self.log),
            logger=self.saver
        )
        self.epochStartCalls = EpochFrequencyHook(
            (self.config.Train.ValFreq, self.validate),
            logger=self.saver
        )

        self.bestRate = 1e10
        self.bestDistortion = -1

        super().__init__(config, modelFn, optimizer, scheduler, saver)

        signal.signal(signal.SIGTERM, self._terminatedHandler)

    def _kill(self):
        sleep(Consts.TimeOut)
        self.saver.critical("[%s] Timeout exceeds, killed.", self.PrettyStep)
        signal.raise_signal(signal.SIGKILL)

    # Handle SIGTERM when main process is terminated.
    # Save necessary info.
    def _terminatedHandler(self, signum, frame):
        killer = threading.Thread(target=self._kill, daemon=True)
        killer.start()
        self.saver.critical("[%s] Main process was interrupted, try to save necessary info.", self.PrettyStep)
        self.saver.critical("[%s] This post-process will be killed after %d secs if stuck.", self.PrettyStep, Consts.TimeOut)
        self.progress.__exit__(None, None, None)
        self.save(os.path.join(self.saver.SaveDir, "last.ckpt"))
        self.saver.critical("[%s] Find the last checkpoint at `%s`", self.PrettyStep, relativePath(os.path.join(self.saver.SaveDir, "last.ckpt")))
        self.summary()
        self.saver.critical("[%s] QUIT.", self.PrettyStep)
        # reset to default SIGTERM handler
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.raise_signal(signal.SIGTERM)

    def summary(self):
        if abs(self.bestRate / self.bestDistortion) > 1e4:
            self.saver.info("[%s] Total epoches: %d, total steps: %s, best rate/distortion: N/A.", self.PrettyStep, self._epoch, self.PrettyStep)
        else:
            self.saver.info("[%s] Total epoches: %d, total steps: %s, best rate/distortion: %.4f / %.2fdB.", self.PrettyStep, self._epoch, self.PrettyStep, self.bestRate, self.bestDistortion)
        self.saver.info("[%s] Test this model by `python -m mcquic.validate --path %s`.", self.PrettyStep, relativePath(os.path.join(self.saver.SaveDir, "[ONE_OF_A].ckpt")))

    def train(self, trainLoader: DataLoader, trainSampler: DistributedSampler, valLoader: DataLoader, *_, beforeRunHook: Optional[Callable] = None, afterRunHook: Optional[Callable] = None, epochStartHook: Optional[Callable] = None, epochFinishHook: Optional[Callable] = None, stepStartHook: Optional[Callable] = None, stepFinishHook: Optional[Callable] = None, **__):
        return super().train(trainLoader, trainSampler,
            beforeRunHook=beforeRunHook,
            afterRunHook=afterRunHook,
            epochStartHook=ChainHook(
                functools.partial(self.epochStartCalls, valLoader=valLoader),
                epochStartHook),
            epochFinishHook=ChainHook(
                self.epochFinishCalls,
                epochFinishHook),
            stepStartHook=stepStartHook,
            stepFinishHook=ChainHook(self._stepFinishHook, stepFinishHook))

    def _beforeRun(self, hook, *args, trainLoader, **kwargs):
        totalBatches = len(trainLoader)
        self.progress.update(self.trainingBar, total=totalBatches)
        self.progress.update(self.epochBar, total=self.config.Train.Epoch * totalBatches, completed=self._step, description=f"[{self._epoch + 1:4d}/{self.config.Train.Epoch:4d}]")
        self.progress.start_task(self.trainingBar)
        self.progress.start_task(self.epochBar)

        super()._beforeRun(hook, *args, totalBatches=totalBatches, **kwargs)
        self.saver.info("[%s] See you at `%s`", self.PrettyStep, self.saver.TensorboardURL)

    def _afterRun(self, hook, *args, **kwArgs):
        self.progress.__exit__(None, None, None)
        super()._afterRun(hook, *args, **kwArgs)
        self.summary()

    def _stepFinishHook(self, *_, rate, distortion, **__):
        distortionDB = self._model.formatDistortion(distortion)
        moment = self.diffTracker(distortionDB)

        task = self.progress.get_task(self.trainingBar)
        self.progress.update(self.trainingBar, advance=1, progress=f"[{task.completed + 1:4d}/{task.total:4d}]", suffix=f"D = [b green]{moment:2.2f}[/]dB")
        self.progress.update(self.epochBar, advance=1)

        if self._step % 100 != 0:
            return
        self.saver.add_scalar(f"Stat/{self.config.Train.Target}", moment, global_step=self._step)
        self.saver.add_scalar("Stat/Lr", self._scheduler.get_last_lr()[0], global_step=self._step)

    def _epochStart(self, hook, *args, **kwArgs):
        self.progress.reset(self.trainingBar)
        self.progress.update(self.epochBar, description=f"[{self._epoch + 1:4d}/{self.config.Train.Epoch:4d}]")
        super()._epochStart(hook, *args, **kwArgs)


    def log(self, *_, images, postProcessed, restored, codes, logits, **__):
        self.saver.add_scalar("Stat/Epoch", self._epoch, self._step)
        # First level, first image, first group
        self.saver.add_histogram("Stat/LogDistance", (-(logits[0][0, 0])).clamp(Consts.Eps).log10(), global_step=self._step)
        freq = self._model.Compressor.NormalizedFreq
        # [m, ki]
        for lv, (fr, c) in enumerate(zip(freq, codes)):
            self.saver.add_histogram_raw(f"Stat/FreqLv{lv}", min=0, max=len(fr[0]), num=len(fr[0]), sum=fr[0].sum(), sum_squares=(fr[0] ** 2).sum(), bucket_limits=list(range(len(fr[0]))), bucket_counts=fr[0], global_step=self._step)
            self.saver.add_images(f"Train/CodeLv{lv}", self.validator.visualizeIntermediate(c), self._step)
        self.saver.add_images("Train/Raw", self.validator.tensorToImage(images), global_step=self._step)
        self.saver.add_images("Train/Post", self.validator.tensorToImage(postProcessed), global_step=self._step)
        self.saver.add_images("Train/Res", self.validator.tensorToImage(restored), global_step=self._step)
        self.saver.add_scalar("Stat/CodeUsage", self._model.Compressor.CodeUsage, global_step=self._step)

    def validate(self, *_, valLoader: DataLoader, **__):
        torch.cuda.empty_cache()

        self.saver.debug("[%s] Start validation at epoch %4d.", self.PrettyStep, self._epoch)

        self._model.eval()
        results, summary = self.validator.validate(self._epoch, self._model.Compressor, valLoader, self.progress)

        self.saver.add_scalar(f"Eval/MsSSIM", results["MsSSIM"], global_step=self._step)
        self.saver.add_scalar(f"Eval/PSNR", results["PSNR"], global_step=self._step)
        self.saver.add_scalar(f"Eval/BPP", results["BPP"], global_step=self._step)
        self.saver.add_images(f"Eval/Visualization", results["Visualization"], global_step=self._step)

        self.save()

        rate, distortion = results["BPP"], results[self.config.Train.Target]

        # TODO: Why d/r continously decrease?
        # if (distortion / rate) > (self.bestDistortion / self.bestRate):
        if distortion > self.bestDistortion:
            self.bestDistortion = distortion
            self.bestRate = rate
            self.progress.update(self.epochBar, suffix=f"H = [b red]{self.bestDistortion:2.2f}[/]dB")
            shutil.copy2(self.saver.SavePath, os.path.join(self.saver.SaveDir, "best.ckpt"))
        self.saver.info("[%s] %s", self.PrettyStep, summary)
        self._model.train()

        self.saver.debug("[%s] End validation at epoch %4d.", self.PrettyStep, self._epoch)


class PalTrainer(_baseTrainer):
    def __init__(self, config: Config, modelFn: Callable[[], Tuple[BaseCompressor, Distortion]], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], saver: Saver):
        if dist.get_rank() == 0:
            raise AttributeError("You should call <MainTrainer> for main process other than <PalTrainer> to save, log necessary information.")
        super().__init__(config, modelFn, optimizer, scheduler, saver)

    def train(self, trainLoader: DataLoader, trainSampler: DistributedSampler, *_, beforeRunHook: Optional[Callable] = None, afterRunHook: Optional[Callable] = None, epochStartHook: Optional[Callable] = None, epochFinishHook: Optional[Callable] = None, stepStartHook: Optional[Callable] = None, stepFinishHook: Optional[Callable] = None, **__):
        return super().train(trainLoader, trainSampler, beforeRunHook=beforeRunHook, afterRunHook=afterRunHook, epochStartHook=epochStartHook, epochFinishHook=epochFinishHook, stepStartHook=stepStartHook, stepFinishHook=stepFinishHook)


def getTrainer(rank: int, config: Config, modelFn: Callable[[], Tuple[BaseCompressor, Distortion]], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], saver: Saver) -> _baseTrainer:
    if rank == 0:
        return MainTrainer(config, modelFn, optimizer, scheduler, saver)
    return PalTrainer(config, modelFn, optimizer, scheduler, saver)
