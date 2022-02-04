import functools
import os
import shutil
from time import sleep
from typing import Callable, List, Optional, Tuple, Type
import signal
import threading

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch import distributed as dist
from vlutils.base.freqHook import ChainHook
from vlutils.saver import Saver
from vlutils.logger import trackingFunctionCalls
from vlutils.base import Restorable
from vlutils.runtime import relativePath
from rich import filesize

from mcquic.consts import Consts
from mcquic.datasets import getTrainingRefLoader
from mcquic.datasets.prefetcher import Prefetcher
from mcquic.evaluation.metrics import Decibel
from mcquic.modules.composed import Composed
from mcquic import Config
from mcquic.modules.compressor import BaseCompressor
from mcquic.train.valueTuners import ValueTuner
from mcquic.utils.helper import EMATracker, EpochFrequencyHook, checkHook, getRichProgress, totalParameters
from mcquic.validate import Validator


class _baseTrainer(Restorable):
    def __init__(self, config: Config, modelFn: Callable[[], Tuple[BaseCompressor, nn.Module]], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], valueTuners: List[Type[ValueTuner]], saver: Saver, **_) -> None:
        super().__init__()
        self.saver = saver

        self.rank = dist.get_rank()
        self.saver.debug("<%s> is located at rank `%d`", self.__class__.__name__, self.rank)
        self.worldSize = dist.get_world_size()
        torch.cuda.set_device(self.rank)
        self.config = config

        self._epoch = 0
        self._step = 0

        self.saver.debug("[%s] Creating model...", self.PrettyStep)
        compressor, criterion = trackingFunctionCalls(modelFn, self.saver)()
        self._model = Composed(compressor.to(self.rank), criterion.to(self.rank), device_ids=[self.rank], output_device=self.rank)
        self.saver.debug("[%s] Model created.", self.PrettyStep)
        self.saver.debug("[%s] Model size: %s", self.PrettyStep, totalParameters(self._model))

        self.saver.debug("[%s] Creating optimizer...", self.PrettyStep)
        optimizer = trackingFunctionCalls(optimizer, self.saver)
        self._optimizer = optimizer(self._model.parameters(), **self.config.Optim.params)
        self.optimFn = optimizer
        self.saver.debug("[%s] Optimizer created.", self.PrettyStep)

        self.saver.debug("[%s] Creating LR scheduler...", self.PrettyStep)
        scheduler = trackingFunctionCalls(scheduler, self.saver)
        self._scheduler = scheduler(self._optimizer, **self.config.Schdr.params)
        self.schdrFn = scheduler
        self.saver.debug("[%s] LR scheduler created.", self.PrettyStep)

        self.saver.debug("[%s] Creating value tuner...", self.PrettyStep)
        self._regularizationTuner = trackingFunctionCalls(valueTuners[0], self.saver)(**self.config.RegSchdr.params)
        self._temperatureTuner = trackingFunctionCalls(valueTuners[1], self.saver)(**self.config.TempSchdr.params)
        self.saver.debug("[%s] Value tuner created.", self.PrettyStep)

        self.saver.debug("[%s] <%s> created.", self.PrettyStep, self.__class__.__name__)

    @property
    def PrettyStep(self):
        unit, suffix = filesize.pick_unit_and_suffix(self._step, [" ", "k", "M"], 1000)
        return f"{(self._step // unit):3d}{suffix}"

    def restoreStates(self, path: str):
        self.saver.debug("[%s] Restored state dict from `%s`", self.PrettyStep, path)

        self.saver.load(path, {"cuda:0": f"cuda:{self.rank}"}, logger=self.saver, **{Consts.Fingerprint: self})

        self.saver.debug("[%s] Restore network parameters finished.", self.PrettyStep)

        self.resetOptimizer()
        for group in self._optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.resetScheduler(self._scheduler.last_epoch) # type: ignore

        self._regularizationTuner._epoch = self._epoch
        self._temperatureTuner._epoch = self._epoch

        self.saver.debug("[%s] Value tuner reset.", self.PrettyStep)

    def resetOptimizer(self):
        del self._optimizer
        self._optimizer = self.optimFn(self._model.parameters(), **self.config.Optim.params)

        self.saver.debug("[%s] Optimizer reset.", self.PrettyStep)

    def resetScheduler(self, lastEpoch=-1):
        del self._scheduler
        self._scheduler = self.schdrFn(self._optimizer, last_epoch=lastEpoch, **self.config.Schdr.params)

        self.saver.debug("[%s] LR scheduler reset.", self.PrettyStep)

    def _beforeRun(self, hook, *args, totalBatches, **kwargs):
        hook(self._step, self._epoch, *args, totalBatches=totalBatches, **kwargs)
        if self._step > 0:
            self.saver.info("[%s] Resume training at %ssteps/%d epochs.", self.PrettyStep, self.PrettyStep, self._epoch)
        else:
            self.saver.info("[%s] Start training.", self.PrettyStep)

        self.saver.debug("[%s] Training loop started.", self.PrettyStep)

    def _afterRun(self, hook, *args, **kwArgs):
        hook(self._step, self._epoch, *args, **kwArgs)
        self.saver.debug("[%s] Training loop finished.", self.PrettyStep)

    def _stepStart(self, hook, *args, **kwArgs):
        hook(self._step, self._epoch, *args, **kwArgs)

    def _stepFinish(self, hook, *args, **kwArgs):
        self._step += 1
        hook(self._step, self._epoch, *args, **kwArgs)

    def _epochStart(self, hook, *args, trainSampler, **kwArgs):
        trainSampler.set_epoch(self._epoch)
        hook(self._step, self._epoch, *args, trainSampler, **kwArgs)

        self.saver.debug("[%s] Epoch %4d started.", self.PrettyStep, self._epoch + 1)

    def _epochFinish(self, hook, *args, trainSet, **kwArgs):
        self._epoch += 1

        self.saver.debug("[%s] Epoch %4d finished.", self.PrettyStep, self._epoch)

        self._scheduler.step()
        self.saver.debug("[%s] Lr is set to %.2e.", self.PrettyStep, self._scheduler.get_last_lr()[0])
        self._regularizationTuner.step()
        self.saver.debug("[%s] Reg is set to %.2e.", self.PrettyStep, self._regularizationTuner.Value)
        self._temperatureTuner.step()
        self.saver.debug("[%s] Temperature is set to %.2e.", self.PrettyStep, self._temperatureTuner.Value)

        hook(self._step, self._epoch, *args, trainSet=trainSet, **kwArgs)

        if self._epoch % (self.config.TestFreq * 10) == 0:
            self.refresh()

    @torch.inference_mode()
    def refresh(self, *_, **__):
        self.saver.debug("[%s] Start refresh at epoch %4d.", self.PrettyStep, self._epoch)

        self._model.eval()
        reAssignProportion = self._model.refresh(self.rank)
        self._model.train()

        self.saver.debug("[%s] %.2f%% of codebook is re-assigned.", self.PrettyStep, reAssignProportion * 100)

        self.saver.debug("[%s] End refresh at epoch %4d.", self.PrettyStep, self._epoch)
        return reAssignProportion

    # def _reduceLoss(self, losses: Tuple[torch.Tensor]) -> torch.Tensor:
    #     return sum(losses)

    def train(self, trainLoader: Prefetcher, trainSampler: DistributedSampler, *_, beforeRunHook: Optional[Callable] = None, afterRunHook: Optional[Callable] = None, epochStartHook: Optional[Callable] = None, epochFinishHook: Optional[Callable] = None, stepStartHook: Optional[Callable] = None, stepFinishHook: Optional[Callable] = None, **__):
        beforeRunHook = checkHook(beforeRunHook, "BeforeRunHook", self.saver)
        afterRunHook = checkHook(afterRunHook, "AfterRunHook", self.saver)
        stepStartHook = checkHook(stepStartHook, "StepStartHook", self.saver)
        stepFinishHook = checkHook(stepFinishHook, "StepFinishHook", self.saver)
        epochStartHook = checkHook(epochStartHook, "EpochStartHook", self.saver)
        epochFinishHook = checkHook(epochFinishHook, "EpochFinishHook", self.saver)

        self._beforeRun(beforeRunHook, totalBatches=len(trainLoader._loader))

        for _ in range(self._epoch, self.config.Epoch):
            self._epochStart(epochStartHook, trainSampler=trainSampler)
            for images in trainLoader:
                self._stepStart(stepStartHook)

                self._optimizer.zero_grad()
                xHat, (rate, distortion), codes, logits = self._model(images, self._temperatureTuner.Value, self._regularizationTuner.Value)
                # loss = self._reduceLoss(losses)
                (rate + distortion).backward()
                self._optimizer.step()

                self._stepFinish(stepFinishHook, rate=rate, distortion=distortion)
            self._epochFinish(epochFinishHook, images=images, restored=xHat, codes=codes, logits=logits, trainSet=trainLoader._loader.dataset) # type: ignore
        self._afterRun(afterRunHook)

class MainTrainer(_baseTrainer):
    def __init__(self, config: Config, modelFn: Callable[[], Tuple[BaseCompressor, nn.Module]], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], valueTuners: List[Type[ValueTuner]], saver: Saver) -> None:
        if dist.get_rank() != 0:
            raise AttributeError("A sub-process should not to be a <MainTrainer>, use <PalTrainer> instead.")

        self.rank = dist.get_rank()
        self.config = config
        self.saver = saver

        self.progress = getRichProgress().__enter__()
        self.trainingBar = self.progress.add_task("", start=False, progress="[----/----]", suffix=Consts.CDot * 10)
        self.epochBar = self.progress.add_task("[----/----]", start=False, progress="", suffix=Consts.CDot * 10)

        self.validator = Validator(self.config, self.rank)

        self.formatter = Decibel(1.0).to(self.rank)
        self.diffTracker = EMATracker(()).to(self.rank)

        hooks = EpochFrequencyHook(
            (1, self.log),
            (self.config.ValFreq, self.validate),
            (self.config.TestFreq, self.test),
            logger=self.saver
        )

        self.epochFinishCalls = hooks

        self.bestDistortion = float("-inf")

        super().__init__(config, modelFn, optimizer, scheduler, valueTuners, saver)

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
        self.saver._savePath = os.path.join(self.saver.SaveDir, "last.ckpt")
        self.saver.save(**{Consts.Fingerprint: self})
        self.saver.critical("[%s] Find the last checkpoint at `%s`", self.PrettyStep, relativePath(self.saver.SavePath))
        self.summary()
        self.saver.critical("[%s] QUIT.", self.PrettyStep)
        # reset to default SIGTERM handler
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.raise_signal(signal.SIGTERM)

    def summary(self):
        self.saver.info("[%s] Total epoches: %d, total steps: %s, best distortion: %.2fdB.", self.PrettyStep, self._epoch, self.PrettyStep, self.bestDistortion)
        self.saver.info("[%s] Test this model by `python -m mcquic.validate --path %s`.", self.PrettyStep, relativePath(os.path.join(self.saver.SaveDir, "[ONE_OF_A].ckpt")))

    def train(self, trainLoader: Prefetcher, trainSampler: DistributedSampler, valLoader: DataLoader, testLoader: DataLoader, *_, beforeRunHook: Optional[Callable] = None, afterRunHook: Optional[Callable] = None, epochStartHook: Optional[Callable] = None, epochFinishHook: Optional[Callable] = None, stepStartHook: Optional[Callable] = None, stepFinishHook: Optional[Callable] = None, **__):
        return super().train(trainLoader, trainSampler,
            beforeRunHook=beforeRunHook,
            afterRunHook=afterRunHook,
            epochStartHook=epochStartHook,
            epochFinishHook=ChainHook(
                functools.partial(self.epochFinishCalls, valLoader=valLoader, testLoader=testLoader),
                epochFinishHook),
            stepStartHook=stepStartHook,
            stepFinishHook=ChainHook(self._stepFinishHook, stepFinishHook))

    def _beforeRun(self, hook, *args, totalBatches, **kwargs):
        self.progress.update(self.trainingBar, total=totalBatches)
        self.progress.update(self.epochBar, total=self.config.Epoch * totalBatches, completed=self._step, description=f"[{self._epoch + 1:4d}/{self.config.Epoch:4d}]")
        self.progress.start_task(self.trainingBar)
        self.progress.start_task(self.epochBar)

        super()._beforeRun(hook, *args, totalBatches=totalBatches, **kwargs)
        self.saver.info("[%s] See you at `%s`", self.PrettyStep, self.saver.TensorboardURL)

    def _afterRun(self, hook, *args, **kwArgs):
        self.progress.__exit__(None, None, None)
        super()._afterRun(hook, *args, **kwArgs)
        self.summary()

    def _stepFinishHook(self, *_, rate, distortion, **__):

        distortionDB = self.formatter(distortion)
        moment = self.diffTracker(distortionDB)

        task = self.progress.get_task(self.trainingBar)
        self.progress.update(self.trainingBar, advance=1, progress=f"[{task.completed + 1:4d}/{task.total:4d}]", suffix=f"D = [b green]{moment:2.2f}[/]dB")
        self.progress.update(self.epochBar, advance=1)

        if self._step % 100 != 0:
            return
        self.saver.add_scalar("Stat/DLoss", distortionDB, global_step=self._step)
        # self._saver.add_scalar("Loss/WeakCodebook", kwArgs["auxiliary"][0], global_step=step)
        # self._saver.add_scalar("Loss/WeakFeature", kwArgs["auxiliary"][1], global_step=step)
        # self._saver.add_scalar("Loss/WeakDiversity", kwArgs["auxiliary"][2], global_step=step)
        # self._saver.add_scalar(_logMapping["predict"], kwArgs["predict"], global_step=step)
        # self._saver.add_scalar(_logMapping["bpp"], kwArgs["bpp"], global_step=step)
        self.saver.add_scalar("Stat/Lr", self._scheduler.get_last_lr()[0], global_step=self._step)
        # self.saver.add_scalar(_logMapping["regCoeff"], self._regularizationTuner.Value, global_step=self._step)
        self.saver.add_scalar("Stat/T", self._temperatureTuner.Value, global_step=self._step)

    def _epochStart(self, hook, *args, **kwArgs):
        self.progress.reset(self.trainingBar)
        self.progress.update(self.epochBar, description=f"[{self._epoch + 1:4d}/{self.config.Epoch:4d}]")
        super()._epochStart(hook, *args, **kwArgs)

    def refresh(self, *_, **__):
        reAssignProportion = super().refresh()
        self.saver.add_scalar("Stat/ReAssignProportion", reAssignProportion, global_step=self._step)

    def log(self, *_, images, restored, codes, logits, **__):
        self.saver.add_scalar("Stat/Epoch", self._epoch, self._step)
        # First level, first image, first group
        self.saver.add_histogram("Stat/LogDistance", (-(logits[0][0, 0])).clamp(Consts.Eps).log10(), global_step=self._step)
        freq = self._model.Compressor.Freq
        # [m, ki]
        for lv, (fr, c) in enumerate(zip(freq, codes)):
            self.saver.add_histogram_raw(f"Stat/FreqLv{lv}", min=0, max=len(fr[0]), num=len(fr[0]), sum=fr[0].sum(), sum_squares=(fr[0] ** 2).sum(), bucket_limits=list(range(len(fr[0]))), bucket_counts=fr[0], global_step=self._step)
            # self.saver.add_histogram(f"Stat/FreqLv{lv}", fr[0], global_step=self._step)
            self.saver.add_images(f"Train/CodeLv{lv}", self.validator.visualizeIntermediate(c), self._step)
        self.saver.add_images("Train/Raw", self.validator.tensorToImage(images), global_step=self._step)
        self.saver.add_images("Train/Res", self.validator.tensorToImage(restored), global_step=self._step)
        self.saver.add_scalar("Stat/CodeUsage", self._model.Compressor.CodeUsage, global_step=self._step)

    def validate(self, *_, valLoader: DataLoader, **__):
        self.saver.debug("[%s] Start validation at epoch %4d.", self.PrettyStep, self._epoch)

        self._model.eval()
        results, summary = self.validator.validate(self._epoch, self._model.Compressor, valLoader, self.progress)

        self.saver.add_scalar(f"Eval/MsSSIM", results["MsSSIM"], global_step=self._step)
        self.saver.add_scalar(f"Eval/PSNR", results["PSNR"], global_step=self._step)
        self.saver.add_scalar(f"Eval/BPP", results["BPP"], global_step=self._step)
        self.saver.add_images(f"Eval/Visualization", results["Visualization"], global_step=self._step)

        self.saver.save(**{Consts.Fingerprint: self})
        if results[self.config.Model.target] > self.bestDistortion:
            self.bestDistortion = results[self.config.Model.target]
            self.progress.update(self.epochBar, suffix=f"H = [b red]{self.bestDistortion:2.2f}[/]dB")
            shutil.copy2(self.saver.SavePath, os.path.join(self.saver.SaveDir, "best.ckpt"))
        self.saver.info("[%4d] %s", self._epoch, summary)
        self._model.train()

        self.saver.debug("[%s] End validation at epoch %4d.", self._epoch)

    def test(self, *_, testLoader: DataLoader, valLoader: DataLoader, **__):
        self.saver.debug("[%s] Start test at epoch %4d.", self.PrettyStep, self._epoch)

        self._model.eval()

        self._model.train()

        self.saver.debug("[%s] End test at epoch %4d.", self.PrettyStep, self._epoch)
        return
        avgRate, avgDistortion = self.validator.validate(self._model.module._compressor, testLoader)

class PalTrainer(_baseTrainer):
    def __init__(self, config: Config, modelFn: Callable[[], Tuple[BaseCompressor, nn.Module]], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], valueTuners: List[Type[ValueTuner]], saver: Saver) -> None:
        if dist.get_rank() == 0:
            raise AttributeError("You should call <MainTrainer> for main process other than <PalTrainer> to save, log necessary information.")
        super().__init__(config, modelFn, optimizer, scheduler, valueTuners, saver)

    def train(self, trainLoader: Prefetcher, trainSampler: DistributedSampler, *_, beforeRunHook: Optional[Callable] = None, afterRunHook: Optional[Callable] = None, epochStartHook: Optional[Callable] = None, epochFinishHook: Optional[Callable] = None, stepStartHook: Optional[Callable] = None, stepFinishHook: Optional[Callable] = None, **__):
        return super().train(trainLoader, trainSampler, beforeRunHook=beforeRunHook, afterRunHook=afterRunHook, epochStartHook=epochStartHook, epochFinishHook=epochFinishHook, stepStartHook=stepStartHook, stepFinishHook=stepFinishHook)


def getTrainer(rank: int, config: Config, model: Callable[[], Tuple[BaseCompressor, nn.Module]], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], valueTuners: List[Type[ValueTuner]], saver: Saver) -> _baseTrainer:
    if rank == 0:
        return MainTrainer(config, model, optimizer, scheduler, valueTuners, saver)
    return PalTrainer(config, model, optimizer, scheduler, valueTuners, saver)
