import functools
import os
import shutil
from typing import Callable, List, Optional, Tuple, Type

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch import distributed as dist
from vlutils.base.freqHook import ChainHook
from vlutils.saver import Saver
from vlutils.base import FrequecyHook
from vlutils.base import Restorable
from rich import filesize

from mcqc.consts import Consts
from mcqc.datasets.dataset import BasicLMDB
from mcqc.datasets import getTrainingRefLoader
from mcqc.datasets.prefetcher import Prefetcher
from mcqc.evaluation.metrics import Decibel
from mcqc.models.composed import Composed
from mcqc import Config
from mcqc.models.compressor import BaseCompressor
from mcqc.training.valueTuners import ValueTuner
from mcqc.utils.helper import EMATracker, getRichProgress, nop
from mcqc.validation import Validator


_logMapping = {
    "distortion": "Loss/Distortion",
    "auxiliary": "Loss/Auxiliary",
    "predict": "Loss/Context",
    "bpp": "Stat/Reg",
    "lr": "Stat/LR",
    "regCoeff": "Stat/Reg",
    "temperature": "Stat/T"
}


class _baseTrainer(Restorable):
    def __init__(self, config: Config, modelFn: Callable[[], Tuple[BaseCompressor, nn.Module]], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], valueTuners: List[Type[ValueTuner]], **_) -> None:
        super().__init__()
        self.rank = dist.get_rank()
        self.worldSize = dist.get_world_size()
        torch.cuda.set_device(self.rank)
        self.config = config

        compressor, criterion = modelFn()

        self._model = Composed(compressor.to(self.rank), criterion.to(self.rank), device_ids=[self.rank], output_device=self.rank)

        self._optimizer = optimizer(self._model.parameters(), **self.config.Optim.params)
        self.optimFn = optimizer
        self._scheduler = scheduler(self._optimizer, **self.config.Schdr.params)
        self.schdrFn = scheduler

        self._regularizationTuner = valueTuners[0](**self.config.RegSchdr.params)
        self._temperatureTuner = valueTuners[1](**self.config.TempSchdr.params)

        self._epoch = 0
        self._step = 0


    def restoreStates(self, ckpt: dict):
        self.load_state_dict(ckpt, False)

        self.resetOptimizer()
        for group in self._optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.resetScheduler(self._scheduler.last_epoch) # type: ignore

        self._regularizationTuner._epoch = self._epoch
        self._temperatureTuner._epoch = self._epoch

    def resetOptimizer(self):
        del self._optimizer
        self._optimizer = self.optimFn(self._model.parameters(), **self.config.Optim.params)

    def resetScheduler(self, lastEpoch=-1):
        del self._scheduler
        self._scheduler = self.schdrFn(self._optimizer, last_epoch=lastEpoch, **self.config.Schdr.params)

    def _beforeRun(self, hook, *args, totalBatches, **kwargs):
        hook(self._step, self._epoch, *args, totalBatches=totalBatches, **kwargs)

    def _afterRun(self, hook, *args, **kwArgs):
        hook(self._step, self._epoch, *args, **kwArgs)

    def _stepStart(self, hook, *args, **kwArgs):
        hook(self._step, self._epoch, *args, **kwArgs)

    def _stepFinish(self, hook, *args, **kwArgs):
        self._step += 1
        hook(self._step, self._epoch, *args, **kwArgs)

    def _epochStart(self, hook, *args, trainSampler, **kwArgs):
        trainSampler.set_epoch(self._epoch)
        hook(self._step, self._epoch, *args, trainSampler, **kwArgs)

    def _epochFinish(self, hook, *args, trainSet, **kwArgs):
        self._scheduler.step()
        self._regularizationTuner.step()
        self._temperatureTuner.step()
        self._epoch += 1
        hook(self._step, self._epoch, *args, **kwArgs)
        if self._epoch % self.config.TestFreq == 0:
            self.refresh()

    @torch.inference_mode()
    def refresh(self, *_, **__):
        self._model.eval()
        self._model.refresh(self.rank)
        self._model.train()

    # def _reduceLoss(self, losses: Tuple[torch.Tensor]) -> torch.Tensor:
    #     return sum(losses)

    def train(self, trainLoader: Prefetcher, trainSampler: DistributedSampler, *_, beforeRunHook: Optional[Callable] = None, afterRunHook: Optional[Callable] = None, epochStartHook: Optional[Callable] = None, epochFinishHook: Optional[Callable] = None, stepStartHook: Optional[Callable] = None, stepFinishHook: Optional[Callable] = None, **__):
        beforeRunHook = beforeRunHook or nop
        afterRunHook = afterRunHook or nop
        stepStartHook = stepStartHook or nop
        stepFinishHook = stepFinishHook or nop
        epochStartHook = epochStartHook or nop
        epochFinishHook = epochFinishHook or nop

        self._beforeRun(beforeRunHook, totalBatches=len(trainLoader._loader))

        for _ in range(self._epoch, self.config.Epoch):
            self._epochStart(epochStartHook, trainSampler=trainSampler)
            for images in trainLoader:
                self._stepStart(stepStartHook)

                self._optimizer.zero_grad()
                xHat, (rate, distortion), codes, logits = self._model(images)
                # loss = self._reduceLoss(losses)
                (rate + distortion).backward()
                self._optimizer.step()

                self._stepFinish(stepFinishHook, rate=rate, distortion=distortion)
                # progress.update(job, advance=1, loss="%2.2fdB" % float(-10 * loss.log10()))
            self._epochFinish(epochStartHook, images=images, restored=xHat, codes=codes, logits=logits, trainSet=trainLoader._loader.dataset) # type: ignore
        self._afterRun(afterRunHook)

class MainTrainer(_baseTrainer):
    def __init__(self, config: Config, modelFn: Callable[[], Tuple[BaseCompressor, nn.Module]], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], valueTuners: List[Type[ValueTuner]], saver: Saver) -> None:
        if dist.get_rank() != 0:
            raise AttributeError("A sub-process should not to be a `MainTrainer`, use `PalTrainer` instead.")
        Restorable.__init__(self)
        self._epoch = 0
        self._step = 0

        self.progress = getRichProgress().__enter__()

        self.trainingBar = self.progress.add_task(Consts.CDot * 6, start=False, progress="preparing", suffix=Consts.CDot * 9)


        self.rank = dist.get_rank()
        self.worldSize = dist.get_world_size()
        torch.cuda.set_device(self.rank)
        self.config = config


        self.saver = saver

        self.validator = Validator(self.rank)

        self.formatter = Decibel(1.0).to(self.rank)
        self.emaTracker = EMATracker(()).to(self.rank)

        self.epochFinishCalls = torch.inference_mode()(FrequecyHook(
            (1, self.log),
            (self.config.ValFreq, self.validate),
            (self.config.TestFreq, self.test)
        ))

        self.bestDistortion = float("-inf")
        self.epochSteps = 0

        compressor, criterion = modelFn()

        self._model = Composed(compressor.to(self.rank), criterion.to(self.rank), device_ids=[self.rank], output_device=self.rank)

        self._optimizer = optimizer(self._model.parameters(), **self.config.Optim.params)
        self.optimFn = optimizer
        self._scheduler = scheduler(self._optimizer, **self.config.Schdr.params)
        self.schdrFn = scheduler

        self._regularizationTuner = valueTuners[0](**self.config.RegSchdr.params)
        self._temperatureTuner = valueTuners[1](**self.config.TempSchdr.params)

    def train(self, trainLoader: Prefetcher, trainSampler: DistributedSampler, valLoader: DataLoader, testLoader: DataLoader, *_, beforeRunHook: Optional[Callable] = None, afterRunHook: Optional[Callable] = None, epochStartHook: Optional[Callable] = None, epochFinishHook: Optional[Callable] = None, stepStartHook: Optional[Callable] = None, stepFinishHook: Optional[Callable] = None, **__):
        self.validate(valLoader=valLoader)
        self.test(testLoader=testLoader, valLoader=valLoader)
        return super().train(trainLoader, trainSampler,
            beforeRunHook=beforeRunHook,
            afterRunHook=afterRunHook,
            epochStartHook=epochStartHook,
            epochFinishHook=ChainHook(
                functools.partial(self.epochFinishCalls, valLoader=valLoader, testLoader=testLoader),
                epochFinishHook),
            stepStartHook=stepStartHook,
            stepFinishHook=ChainHook(self._stepFinishHook, stepFinishHook))

    def restoreStates(self, ckpt: dict):
        self.saver.info("Restore state dict from %s", os.path.relpath(self.saver.SavePath))
        return super().restoreStates(ckpt)

    def _beforeRun(self, hook, *args, totalBatches, **kwargs):
        self.progress.update(self.trainingBar, total=totalBatches)
        self.epochSteps = totalBatches
        self.progress.start_task(self.trainingBar)

        super()._beforeRun(hook, *args, totalBatches=totalBatches, **kwargs)

        if self._step > 0:
            self.saver.info("Resume training at %s/%d epochs.", self.prettyStep, self._epoch)
        else:
            self.saver.info("Start training.")
        self.saver.info("See you at %s", self.saver.TensorboardURL)

    @property
    def prettyStep(self):
        unit, suffix = filesize.pick_unit_and_suffix(self._step, [" steps", "k steps", "M steps"], 1000)
        return f"{(self._step / float(unit)):3.2g}{suffix}"

    def _afterRun(self, hook, *args, **kwArgs):
        self.progress.__exit__(None, None, None)
        super()._afterRun(hook, *args, **kwArgs)

    def _stepFinishHook(self, *_, rate, distortion, **__):
        self.progress.update(self.trainingBar, advance=1, progress=f"{(self._step % self.epochSteps):4d}/{self.epochSteps:4d}")
        emaDistortion = self.emaTracker(distortion)
        if (distortion - emaDistortion).abs() > emaDistortion * 0.05:
            self.progress.update(self.trainingBar, suffix=f"D = [b red]{self.formatter(emaDistortion):2.2f}[/]dB")
        if self._step % 100 != 0:
            return
        self.saver.add_scalar("Loss/Distortion", self.formatter(distortion), global_step=self._step)
        # self._saver.add_scalar("Loss/WeakCodebook", kwArgs["auxiliary"][0], global_step=step)
        # self._saver.add_scalar("Loss/WeakFeature", kwArgs["auxiliary"][1], global_step=step)
        # self._saver.add_scalar("Loss/WeakDiversity", kwArgs["auxiliary"][2], global_step=step)
        # self._saver.add_scalar(_logMapping["predict"], kwArgs["predict"], global_step=step)
        # self._saver.add_scalar(_logMapping["bpp"], kwArgs["bpp"], global_step=step)
        self.saver.add_scalar(_logMapping["lr"], self._scheduler.get_last_lr()[0], global_step=self._step)
        self.saver.add_scalar(_logMapping["regCoeff"], self._regularizationTuner.Value, global_step=self._step)
        self.saver.add_scalar(_logMapping["temperature"], self._temperatureTuner.Value, global_step=self._step)

    def _epochStart(self, hook, *args, **kwArgs):
        self.progress.reset(self.trainingBar)
        self.progress.update(self.trainingBar, description=f"#{(self._epoch + 1):4d}")
        super()._epochStart(hook, *args, **kwArgs)

    def log(self, *_, images, restored, codes, logits, **__):
        self.saver.add_scalar("Stat/Epoch", self._epoch, self._step)
        self.saver.add_histogram("Stat/Logit", logits[0][0, 0], global_step=self._step)
        for i, c in enumerate(codes):
            self.saver.add_histogram(f"Stat/Code{i}", c[0, 0].flatten(), global_step=self._step)
            self.saver.add_images(f"Train/Code{i}", self.validator.visualizeIntermediate(c), self._step)
        self.saver.add_images("Train/Raw", self.validator.tensorToImage(images), global_step=self._step)
        self.saver.add_images("Train/Res", self.validator.tensorToImage(restored), global_step=self._step)

    def validate(self, *_, valLoader: DataLoader, **__):
        self._model.eval()
        results, summary = self.validator.validate(self._epoch, self._model.Compressor, valLoader, self.progress)
        self.saver.save(**{Consts.Fingerprint: self})
        if results[self.config.Model.target] > self.bestDistortion:
            self.bestDistortion = results[self.config.Model.target]
            shutil.copy2(self.saver.SavePath, os.path.join(self.saver.SaveDir, "best.ckpt"))
        self.saver.info("[%4d] " + ", ".join([f"{key}: {value}" for key, value in summary.items()]), self._epoch)
        self._model.train()

    def test(self, *_, testLoader: DataLoader, valLoader: DataLoader, **__):
        self._model.eval()

        self._model.Compressor.clearFreq()
        refLoader = getTrainingRefLoader(self.config.Dataset, self.config.BatchSize)
        self.validator.count(self._epoch, self._model.Compressor, valLoader, self.progress)

        self._model.train()
        return
        avgRate, avgDistortion = self.validator.validate(self._model.module._compressor, testLoader)

class PalTrainer(_baseTrainer):
    def __init__(self, config: Config, modelFn: Callable[[], Tuple[BaseCompressor, nn.Module]], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], valueTuners: List[Type[ValueTuner]], **_) -> None:
        if dist.get_rank() == 0:
            raise AttributeError("You should call `MainTrainer` for main process other than `PalTrainer` to save, log necessary information.")
        super().__init__(config, modelFn, optimizer, scheduler, valueTuners)

    def train(self, trainLoader: Prefetcher, trainSampler: DistributedSampler, *_, beforeRunHook: Optional[Callable] = None, afterRunHook: Optional[Callable] = None, epochStartHook: Optional[Callable] = None, epochFinishHook: Optional[Callable] = None, stepStartHook: Optional[Callable] = None, stepFinishHook: Optional[Callable] = None, **__):
        return super().train(trainLoader, trainSampler, beforeRunHook=beforeRunHook, afterRunHook=afterRunHook, epochStartHook=epochStartHook, epochFinishHook=epochFinishHook, stepStartHook=stepStartHook, stepFinishHook=stepFinishHook)


def getTrainer(rank: int, config: Config, model: Callable[[], Tuple[BaseCompressor, nn.Module]], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], valueTuners: List[Type[ValueTuner]], saver: Saver) -> _baseTrainer:
    if rank == 0:
        return MainTrainer(config, model, optimizer, scheduler, valueTuners, saver)
    return PalTrainer(config, model, optimizer, scheduler, valueTuners)
