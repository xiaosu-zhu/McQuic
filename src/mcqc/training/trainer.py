import functools
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from vlutils.base.freqHook import ChainHook
from vlutils.config import summary
from vlutils.saver import Saver
from vlutils.base import FrequecyHook
from vlutils.base import Restorable

from mcqc.consts import Consts
from mcqc.datasets.helper import getTestSet, getTrainingSet, getValidationSet
from mcqc.datasets.prefetcher import Prefetcher
from mcqc.evaluation.metrics import Decibel
from mcqc.models.composed import Composed
from mcqc import Config
from mcqc.models.compressor import Compressor
from mcqc.loss import CompressionLossBig
from mcqc.training.valueTuners import ValueTuner
from mcqc.utils.helper import EMATracker, getRichProgress, getSaver, initializeProcessGroup, nop
from mcqc.utils.registry import LrSchedulerRegistry, OptimizerRegistry, ValueTunerRegistry
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
    def __init__(self, config: Config, modelFn: Callable[[], Composed], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], valueTuners: List[Type[ValueTuner]], **_) -> None:
        super().__init__()
        self.rank = dist.get_rank()
        self.worldSize = dist.get_world_size()
        torch.cuda.set_device(self.rank)
        self.config = config

        model = modelFn()

        self._model = DistributedDataParallel(model.to(self.rank), device_ids=[self.rank], output_device=self.rank)

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

    def _epochFinish(self, hook, *args, **kwArgs):
        self._scheduler.step()
        self._regularizationTuner.step()
        self._temperatureTuner.step()
        self._epoch += 1
        hook(self._step, self._epoch, *args, **kwArgs)

    # def _reduceLoss(self, losses: Tuple[torch.Tensor]) -> torch.Tensor:
    #     return sum(losses)

    def train(self, trainLoader: Prefetcher, trainSampler: DistributedSampler, *_, beforeRunHook: Optional[Callable] = None, afterRunHook: Optional[Callable] = None, epochStartHook: Optional[Callable] = None, epochFinishHook: Optional[Callable] = None, stepStartHook: Optional[Callable] = None, stepFinishHook: Optional[Callable] = None, **__):
        beforeRunHook = beforeRunHook or nop
        afterRunHook = afterRunHook or nop
        stepStartHook = stepStartHook or nop
        stepFinishHook = stepFinishHook or nop
        epochStartHook = epochStartHook or nop
        epochFinishHook = epochFinishHook or nop

        totalBatches = len(trainLoader._loader.dataset) // (self.config.BatchSize * self.worldSize) + 1 # type: ignore

        self._beforeRun(beforeRunHook, totalBatches=totalBatches)

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
            self._epochFinish(epochStartHook, images=images, restored=xHat, codes=codes, logits=logits) # type: ignore
        self._afterRun(afterRunHook)

class MainTrainer(_baseTrainer):
    def __init__(self, config: Config, modelFn: Callable[[], Composed], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], valueTuners: List[Type[ValueTuner]], saver: Saver) -> None:
        if dist.get_rank() != 0:
            raise AttributeError("A sub-process should not to be a `MainTrainer`, use `PalTrainer` instead.")
        super(_baseTrainer).__init__()
        self._epoch = 0
        self._step = 0

        self.progress = getRichProgress().__enter__()

        self.trainingBar = self.progress.add_task("training", start=False, epoch=self._epoch + 1, loss=0.0)


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
            (self.config.TestFreq, self.test),
            (self.config.TestFreq, self.reSpread)
        ))

        self.bestDistortion = float("-inf")


        model = modelFn()
        self._model = DistributedDataParallel(model.to(self.rank), device_ids=[self.rank], output_device=self.rank)

        self._optimizer = optimizer(self._model.parameters(), **self.config.Optim.params)
        self.optimFn = optimizer
        self._scheduler = scheduler(self._optimizer, **self.config.Schdr.params)
        self.schdrFn = scheduler

        self._regularizationTuner = valueTuners[0](**self.config.RegSchdr.params)
        self._temperatureTuner = valueTuners[1](**self.config.TempSchdr.params)

    def train(self, trainLoader: Prefetcher, trainSampler: DistributedSampler, valLoader: DataLoader, testLoader: DataLoader, *_, beforeRunHook: Optional[Callable] = None, afterRunHook: Optional[Callable] = None, epochStartHook: Optional[Callable] = None, epochFinishHook: Optional[Callable] = None, stepStartHook: Optional[Callable] = None, stepFinishHook: Optional[Callable] = None, **__):
        return super().train(trainLoader, trainSampler,
            beforeRunHook=beforeRunHook,
            afterRunHook=afterRunHook,
            epochStartHook=epochStartHook,
            epochFinishHook=ChainHook(
                functools.partial(self.epochFinishCalls, trainLoader=trainLoader, valLoader=valLoader, testLoader=testLoader),
                epochFinishHook),
            stepStartHook=stepStartHook,
            stepFinishHook=ChainHook(self._stepFinishHook, stepFinishHook))

    def restoreStates(self, ckpt: dict):
        self.saver.info("Restore state dict from %s", os.path.relpath(self.saver.SavePath))
        return super().restoreStates(ckpt)

    def _beforeRun(self, hook, *args, totalBatches, **kwargs):
        self.progress.update(self.trainingBar, total=totalBatches)
        self.progress.start_task(self.trainingBar)

        super()._beforeRun(hook, *args, totalBatches=totalBatches, **kwargs)

        if self._step > 0:
            self.saver.info("Resume training at %dk steps/%d epochs.", self._step // 1000, self._epoch)
        else:
            self.saver.info("Start training.")
        self.saver.info("See you at %s", self.saver.TensorboardURL)

    def _afterRun(self, hook, *args, **kwArgs):
        self.progress.__exit__(None, None, None)
        super()._afterRun(hook, *args, **kwArgs)

    def _stepFinishHook(self, *_, rate, distortion, **__):
        self.progress.advance(self.trainingBar)
        emaDistortion = self.emaTracker(distortion)
        if (distortion - emaDistortion).abs() > emaDistortion * 0.05:
            self.progress.update(self.trainingBar, loss=self.formatter(emaDistortion))
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

    def _epochStart(self, hook, *args, trainSampler, **kwArgs):
        self.progress.update(self.trainingBar, completed=0, epoch=self._epoch + 1)
        super()._epochStart(hook, *args, trainSampler=trainSampler, **kwArgs)

    def log(self, *_, images, restored, codes, logits, **__):
        self.saver.add_scalar("Stat/Epoch", self._epoch, self._step)
        self.saver.add_histogram("Stat/Logit", logits[0][0, 0], global_step=self._step)
        for i, c in enumerate(codes):
            self.saver.add_histogram(f"Stat/Code{i}", c[0, 0].flatten(), global_step=self._step)
            self.saver.add_images(f"Train/Code{i}", self.validator.visualizeIntermediate(c), self._step)
        self.saver.add_images("Train/Raw", self.validator.tensorToImage(images), global_step=self._step)
        self.saver.add_images("Train/Res", self.validator.tensorToImage(restored), global_step=self._step)

    def validate(self, *_, trainLoader: DataLoader, valLoader: DataLoader, **__):
        self.validator.count(self._epoch, self._model.module._compressor, trainLoader) # type: ignore
        results, summary = self.validator.validate(self._epoch, self._model.module._compressor, valLoader) # type: ignore
        self.saver.save(**{Consts.Fingerprint: self})
        if results[self.config.Model.target] > self._bestDistortion:
            self._bestDistortion = results[self.config.Model.target]
            shutil.copy2(self.saver.SavePath, os.path.join(self.saver.SaveDir, "best.ckpt"))
        self.saver.info("[%04d]" + ", ".join([f"{key}: {value}" for key, value in summary.items()]), self._epoch)

    def reSpread(self, *_, trainLoader: DataLoader, **__):
        pass

    def test(self, *_, testLoader: DataLoader, **__):
        return
        avgRate, avgDistortion = self.validator.validate(self._model.module._compressor, testLoader)

class PalTrainer(_baseTrainer):
    def __init__(self, config: Config, modelFn: Callable[[], Composed], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], valueTuners: List[Type[ValueTuner]], **_) -> None:
        if dist.get_rank() == 0:
            raise AttributeError("You should call `MainTrainer` for main process other than `PalTrainer` to save, log necessary information.")
        super().__init__(config, modelFn, optimizer, scheduler, valueTuners)

    def train(self, trainLoader: Prefetcher, trainSampler: DistributedSampler, *_, beforeRunHook: Optional[Callable] = None, afterRunHook: Optional[Callable] = None, epochStartHook: Optional[Callable] = None, epochFinishHook: Optional[Callable] = None, stepStartHook: Optional[Callable] = None, stepFinishHook: Optional[Callable] = None, **__):
        return super().train(trainLoader, trainSampler, beforeRunHook=beforeRunHook, afterRunHook=afterRunHook, epochStartHook=epochStartHook, epochFinishHook=epochFinishHook, stepStartHook=stepStartHook, stepFinishHook=stepFinishHook)

def getTrainer(rank: int, config: Config, model: Callable[[], Composed], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], valueTuners: List[Type[ValueTuner]], saver: Saver) -> _baseTrainer:
    if rank == 0:
        return MainTrainer(config, model, optimizer, scheduler, valueTuners, saver)
    return PalTrainer(config, model, optimizer, scheduler, valueTuners)

def modelFn(channel, m, k, lossTarget) -> Composed:
    compressor = Compressor(channel, m, k)
    # compressor = PQCompressorBig(config.Model.m, config.Model.k, config.Model.channel, False, False, False, False, -1)
    # print(sum([p.numel() for p in compressor.parameters()]))
    # exit()
    criterion = CompressionLossBig(lossTarget)

    return Composed(compressor, criterion)

def train(rank: int, worldSize: int, port: str, config: Config, saveDir: str, continueTrain: bool, debug: bool):
    saver = getSaver(saveDir, saveName="saved.ckpt", loggerName=Consts.Fingerprint, loggingLevel="DEBUG" if debug else "INFO", config=config, reserve=continueTrain, disable=rank != 0)

    saver.info(summary(config))

    saver.info("Create trainer...")

    initializeProcessGroup(port, rank, worldSize)


    optimizerFn = OptimizerRegistry.get("Lamb")
    schdrFn = LrSchedulerRegistry.get(config.Schdr.type)
    valueTunerFns = [ValueTunerRegistry.get(config.RegSchdr.type), ValueTunerRegistry.get(config.TempSchdr.type)]

    trainer = getTrainer(rank, config, lambda: modelFn(config.Model.channel, config.Model.m, config.Model.k, config.Model.target), optimizerFn, schdrFn, valueTunerFns, saver=saver)

    trainingSet, trainSampler = getTrainingSet(rank, worldSize, config.Dataset, config.BatchSize)
    validationSet = getValidationSet(config.ValDataset, config.BatchSize, disable=rank != 0)
    testSet = getTestSet(config.ValDataset, disable=rank != 0)


    if continueTrain:
        trainer.restoreStates(torch.load(saver.SavePath, {"cuda:0": f"cuda:{rank}"})[Consts.Fingerprint])
    trainer.train(trainingSet, trainSampler, validationSet, testSet)
