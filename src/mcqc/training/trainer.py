import functools
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from vlutils.config import summary
from vlutils.saver import DummySaver, Saver
from vlutils.base import FrequecyHook
from vlutils.base import Restorable
from tqdm.contrib.logging import logging_redirect_tqdm

from mcqc.consts import Consts
from mcqc.datasets.dataset import Basic, BasicLMDB
from mcqc.datasets.prefetcher import Prefetcher
from mcqc.evaluation.metrics import Decibel
from mcqc.models.composed import Composed
from mcqc import Config
from mcqc.models.compressor import Compressor
from mcqc.loss import CompressionLossBig
from mcqc.training.valueTuners import ValueTuner
from mcqc.utils.helper import EMATracker, getRichProgress, initializeProcessGroup
from mcqc.utils.registry import LrSchedulerRegistry, OptimizerRegistry, ValueTunerRegistry
from mcqc.utils.vision import getEvalTransform, getTestTransform, getTrainingPreprocess, getTrainingTransform
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
    def __init__(self, config: Config, model: Composed, optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], valueTuners: List[Type[ValueTuner]]) -> None:
        super().__init__()
        self.rank = dist.get_rank()
        self.worldSize = dist.get_world_size()
        torch.cuda.set_device(self.rank)
        self.config = config

        self.progress = getRichProgress().__enter__()
        self.trainingBar = self.progress.add_task("training", start=False, epoch=self._epoch + 1, loss="       ")

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

        self.resetScheduler(self._scheduler.last_epoch)

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

    def _beforeRunHook(self, step, epoch, *args, **kwArgs):
        pass

    def _afterRun(self, hook, *args, **kwArgs):
        self.progress.__exit__(None, None, None)
        hook(self._step, self._epoch, *args, **kwArgs)

    def _afterRunHook(self, step, epoch, *args, **kwArgs):
        pass

    def _stepStart(self, hook, *args, **kwArgs):
        hook(self._step, self._epoch, *args, **kwArgs)

    def _stepStartHook(self, step, epoch, *args, **kwArgs):
        pass

    def _stepFinish(self, hook, *args, **kwArgs):
        self._step += 1
        hook(self._step, self._epoch, *args, **kwArgs)

    def _stepFinishHook(self, step, epoch, *args, **kwArgs):
        pass

    def _epochStart(self, hook, *args, trainSampler, **kwArgs):
        trainSampler.set_epoch(self._epoch)
        hook(self._step, self._epoch, *args, trainSampler, **kwArgs)

    def _epochStartHook(self, step, epoch, *args, **kwArgs):
        pass

    def _epochFinish(self, hook, *args, **kwArgs):
        self._scheduler.step()
        self._regularizationTuner.step()
        self._temperatureTuner.step()
        self._epoch += 1
        hook(self._step, self._epoch, *args, **kwArgs)

    def _epochFinishHook(self, step, epoch, *args, **kwArgs):
        pass

    def _reduceLoss(self, losses: Tuple[torch.Tensor]) -> torch.Tensor:
        return sum(losses)

    def train(self, trainLoader: Prefetcher, trainSampler: DistributedSampler, beforeRunHook: Optional[Callable] = None, afterRunHook: Optional[Callable] = None, epochStartHook: Optional[Callable] = None, epochFinishHook: Optional[Callable] = None, stepStartHook: Optional[Callable] = None, stepFinishHook: Optional[Callable] = None):
        if beforeRunHook is None:
            beforeRunHook = self._beforeRunHook
        if afterRunHook is None:
            afterRunHook = self._afterRunHook
        if stepStartHook is None:
            stepStartHook = self._stepStartHook
        if stepFinishHook is None:
            stepFinishHook = self._stepFinishHook
        if epochStartHook is None:
            epochStartHook = self._epochStartHook
        if epochFinishHook is None:
            epochFinishHook = self._epochFinishHook

        totalBatches = len(trainLoader._loader.dataset) // (self.config.BatchSize * self.worldSize) + 1 # type: ignore

        self._beforeRun(self._step, self._epoch, beforeRunHook, totalBatches=totalBatches)

        images: Union[torch.Tensor, None] = None
        xHat: Union[torch.Tensor, None] = None
        stats: Dict[str, Any] = {}

        for _ in range(self._epoch, self.config.Epoch):
            self._epochStart(self._step, self._epoch, epochStartHook, trainSampler=trainSampler)
            for images in trainLoader:
                self._stepStartHook(self._step, self._epoch, stepStartHook)

                self._optimizer.zero_grad()
                xHat, loss, stats = self._model(images)
                # loss = self._reduceLoss(losses)
                loss.backward()
                self._optimizer.step()

                self._stepFinish(self._step, self._epoch, stepFinishHook, loss=[loss])
                # progress.update(job, advance=1, loss="%2.2fdB" % float(-10 * loss.log10()))
            self._epochFinish(epochStartHook, images=images, restored=xHat, **stats)
        self._afterRun(afterRunHook)

class MainTrainer(_baseTrainer):
    def __init__(self, config: Config, model: Composed, optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], valueTuners: List[Type[ValueTuner]], saver: Saver) -> None:
        if dist.get_rank() != 0:
            raise AttributeError("A sub-process should not to be a `MainTrainer`, use `PalTrainer` instead.")

        super().__init__(config, model, optimizer, scheduler, valueTuners)

        self.saver = saver

        self.validator = Validator(self.rank)

        self.formatter = Decibel(1.0).to(self.rank)
        self.emaTracker = EMATracker(()).to(self.rank)

        self.epochFinishCalls = FrequecyHook(
            (1, self.log),
            (self.config.ValFreq, self.validate),
            (self.config.TestFreq, self.test),
            (self.config.TestFreq, self.reSpread)
        )

        self._bestDistortion = float("-inf")

    def train(self, trainLoader: Prefetcher, trainSampler: DistributedSampler, valLoader: DataLoader, testLoader: DataLoader):
        return super().train(trainLoader, trainSampler,
            self._beforeRunHook,
            self._stepFinishHook,
            functools.partial(self._epochFinishHook, trainLoader=trainLoader, valLoader=valLoader, testLoader=testLoader),
            self._afterRunHook)

    def restoreStates(self, ckpt: dict):
        self.saver.info("Restore state dict from %s", os.path.relpath(self.saver.SavePath))
        return super().restoreStates(ckpt)

    def _beforeRun(self, hook, *args, totalBatches, **kwargs):
        self.progress.update(self.trainingBar, total=totalBatches)
        self.progress.start_task(self.trainingBar)

        super()._beforeRun(hook, *args, totalBatches, **kwargs)

        if self._step > 0:
            self.saver.info("Resume training at %dk steps/%d epochs.", self._step // 1000, self._epoch)
        else:
            self.saver.info("Start training.")
        self.saver.info("See you at %s", self.saver.TensorboardURL)

    @ torch.inference_mode()
    def _stepFinishHook(self, *, loss, **_):
        self.progress.advance(self.trainingBar)
        distortion = loss[0]
        emaDistortion = self.emaTracker(distortion)
        if (distortion - emaDistortion).abs() > emaDistortion * 0.1:
            self.progress.update(self.trainingBar, loss="%2.2fdB" % self.formatter(distortion))
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
        super()._epochStart(hook, *args, trainSampler, **kwArgs)

    @torch.inference_mode()
    def _epochFinishHook(self, step, epoch, *args, **kwArgs):
        self.saver.add_scalar("Stat/Epoch", epoch, step)
        self.epochFinishCalls(epoch, step, *args, **kwArgs)

    def log(self, epoch, step, *, images, restored, logits, codes, **_):
        self.saver.add_histogram("Stat/Logit", logits[0][0, 0], global_step=step)
        for i, c in enumerate(codes):
            self.saver.add_histogram(f"Stat/Code{i}", c[0, 0].flatten(), global_step=step)
            self.saver.add_images(f"Train/Code{i}", self.validator.visualizeIntermediate(c), step)
        self.saver.add_images("Train/Raw", self.validator.tensorToImage(images), global_step=step)
        self.saver.add_images("Train/Res", self.validator.tensorToImage(restored), global_step=step)

    def validate(self, epoch, step, *, trainLoader: DataLoader, valLoader: DataLoader, **_):
        self.validator.count(epoch, self._model.module._compressor, trainLoader)
        results, summary = self.validator.validate(epoch, self._model.module._compressor, valLoader)
        self.saver.save(**{Consts.Fingerprint: self})
        if results[self.config.Model.target] > self._bestDistortion:
            self._bestDistortion = results[self.config.Model.target]
            shutil.copy2(self.saver.SavePath, os.path.join(self.saver.SaveDir, "best.ckpt"))
        self.saver.info("[%04d]" + ", ".join([f"{key}: {value}" for key, value in summary.items()]), epoch)

    def reSpread(self, *, trainLoader: DataLoader, **_):
        pass

    def test(self, epoch, step, *, testLoader: DataLoader, **_):
        return
        avgRate, avgDistortion = self.validator.validate(self._model.module._compressor, testLoader)

class PalTrainer(_baseTrainer):
    def __init__(self, config: Config, model: Composed, optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], valueTuners: List[Type[ValueTuner]]) -> None:
        if dist.get_rank() == 0:
            raise AttributeError("You should call `MainTrainer` for main process other than `PalTrainer` to save, log necessary information.")
        super().__init__(config, model, optimizer, scheduler, valueTuners)

    def train(self, trainLoader: Prefetcher, trainSampler: DistributedSampler, *_):
        return super().train(trainLoader, trainSampler)

def train(rank: int, worldSize: int, port: str, config: Config, saveDir: str, continueTrain: bool, debug: bool):
    if rank == 0:
        saverFn = Saver
    else:
        saverFn = DummySaver
    saver = saverFn(saveDir, saveName="saved.ckpt", loggerName=Consts.Fingerprint, loggingLevel="DEBUG" if debug else "INFO", config=config, reserve=continueTrain)

    saver.info(summary(config))

    saver.info("Create trainer...")

    initializeProcessGroup(port, rank, worldSize)

    compressor = Compressor(config.Model.channel, config.Model.m, config.Model.k)
    # compressor = PQCompressorBig(config.Model.m, config.Model.k, config.Model.channel, False, False, False, False, -1)
    # print(sum([p.numel() for p in compressor.parameters()]))
    # exit()
    criterion = CompressionLossBig(config.Model.target)

    model = Composed(compressor, criterion)

    optimizerFn = OptimizerRegistry.get("Lamb")
    schdrFn = LrSchedulerRegistry.get(config.Schdr.type)
    valueTunerFns = (ValueTunerRegistry.get(config.RegSchdr.type), ValueTunerRegistry.get(config.TempSchdr.type))\

    trainDataset = BasicLMDB(os.path.join("data", config.Dataset), maxTxns=(config.BatchSize + 4) * worldSize, repeat=config.Repeat, transform=getTrainingPreprocess())
    trainSampler = DistributedSampler(trainDataset, worldSize, rank)
    trainLoader = DataLoader(trainDataset, sampler=trainSampler, batch_size=min(config.BatchSize, len(trainDataset)), num_workers=config.BatchSize + 4, pin_memory=True, persistent_workers=True)
    prefetcher = Prefetcher(trainLoader, rank, getTrainingTransform())

    if rank == 0:
        valDataset = Basic(os.path.join("data", config.ValDataset), transform=getEvalTransform())
        valLoader = DataLoader(valDataset, batch_size=min(config.BatchSize, len(valDataset)), shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

        testDataset = Basic(os.path.join("data", config.ValDataset), transform=getTestTransform())
        testLoader = DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

        trainer = MainTrainer(config, model, optimizerFn, schdrFn, valueTunerFns, saver)
        if continueTrain:
            trainer.restoreStates(torch.load(saver.SavePath, {"cuda:0": f"cuda:{rank}"})[Consts.Fingerprint])
        trainer.train(prefetcher, trainSampler, valLoader, testLoader)
    else:
        trainer = PalTrainer(config, model, optimizerFn, schdrFn, valueTunerFns)
        if continueTrain:
            trainer.restoreStates(torch.load(saver.SavePath, {"cuda:0": f"cuda:{rank}"})[Consts.Fingerprint])
        trainer.train(prefetcher, trainSampler)
