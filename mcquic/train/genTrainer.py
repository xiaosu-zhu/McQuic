import functools
import os
import pathlib
import shutil
from time import sleep
from typing import Callable, Optional, Tuple, Type, Dict, Any
import signal
import threading
import gc
import wandb
import numpy as np
import datetime

import torch
from torchvision.transforms.functional import to_pil_image
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from vlutils.base.freqHook import ChainHook
from vlutils.logger import trackingFunctionCalls
from vlutils.base import Restorable, FrequecyHook
from vlutils.runtime import relativePath
from rich import filesize
import torch.nn.functional as F

from mcquic.train.utils import Saver
from mcquic.consts import Consts
from mcquic.loss import Distortion
from mcquic.validate.utils import EMATracker
from mcquic.modules.compound import Compound
from mcquic import Config
from mcquic.modules.compressor import BaseCompressor
from mcquic.validate import Validator
from mcquic.utils import StrPath, totalParameters
from mcquic.datasets.transforms import getTrainingTransform

from mcquic.train.utils import checkHook, getRichProgress

class _baseGenTrainer(Restorable):
    def __init__(self, config: Config, tmpFile: Optional[StrPath], modelFn: Callable[[], Tuple[BaseCompressor, Distortion]], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], saver: Saver, **_):
        super().__init__()
        self.saver = saver

        self.rank = dist.get_rank()

        self.saver.debug("<%s> is located at rank `%d`", self.__class__.__name__, self.rank)
        self.worldSize = dist.get_world_size()
        localRank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(localRank)
        self.config = config

        self._step = 0

        self._totalStep = config.Train.TotalStep

        # Used for self.PrettyStep
        self.lastFormatted = -1
        self.prettyStep = "......"

        self.transform = getTrainingTransform().to(localRank)

        self.saver.debug("[%s] Creating model...", self.PrettyStep)
        generator = trackingFunctionCalls(modelFn, self.saver)()


        self._model = DistributedDataParallel(generator.to(localRank), device_ids=[localRank], output_device=localRank, find_unused_parameters=False)
        self.saver.debug("[%s] Model created.", self.PrettyStep)
        self.saver.info("[%s] Model size: %s", self.PrettyStep, totalParameters(self._model))

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

        self.tmpFile = tmpFile

        self.saver.debug("[%s] <%s> created.", self.PrettyStep, self.__class__.__name__)

    def periodicSave(self, *_, **__):
        if int(os.environ['LOCAL_RANK']) == 0:
            self.save()

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

    def restoreStates(self, path: StrPath):
        self.saver.debug("[%s] Restored state dict from `%s`", self.PrettyStep, path)

        try:
            self.saver.load(path, torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}"), logger=self.saver, trainer=self)
        except RuntimeError:
            oldCkpt = torch.load(path, "cpu")
            # Using a finetune config
            if self.config.Model.Params["m"] != oldCkpt["config"]["model"]["params"]["m"]:
                pass
            else:
                raise

        self.saver.debug("[%s] Restore network parameters finished.", self.PrettyStep)

        self.resetOptimizer()

        self.resetScheduler(self._scheduler.last_epoch)

    def resetOptimizer(self):
        del self._optimizer
        self._optimizer = self.optimFn(self._model.parameters(), **self.config.Train.Optim.Params)

        for group in self._optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.saver.debug("[%s] Optimizer reset.", self.PrettyStep)

    def resetScheduler(self, lastEpoch=-1):
        del self._scheduler
        self._scheduler = self.schdrFn(self._optimizer, last_epoch=lastEpoch, **self.config.Train.Schdr.Params)

        self.saver.debug("[%s] LR scheduler reset.", self.PrettyStep)

    def _beforeRun(self, hook, *args, **kwArgs):
        if self.tmpFile is not None:
            self.restoreStates(self.tmpFile)
            self.saver.info("[%s] Resume training at %s steps.", self.PrettyStep, self.PrettyStep)
        self.saver.info("[%s] Start training.", self.PrettyStep)

        hook(self._step, 0, self, *args, logger=self.saver, **kwArgs)

    def _afterRun(self, hook, *args, **kwArgs):
        self.saver.debug("[%s] Training loop finished.", self.PrettyStep)
        hook(self._step, 0, self, *args, logger=self.saver, **kwArgs)

    def _stepStart(self, hook, *args, **kwArgs):
        hook(self._step, 0, self, *args, logger=self.saver, **kwArgs)
        self.saver.debug("[%s] Call `stepStart` hooks done.", self.PrettyStep)

    def _stepFinish(self, hook, *args, **kwArgs):
        self._step += 1
        if self._step > 10:
            self._scheduler.step()
        hook(self._step, 0, self, *args, logger=self.saver, **kwArgs)
        self.saver.debug("[%s] Call `stepFinish` hooks done.", self.PrettyStep)

    # def _epochStart(self, hook, *args, **kwArgs):
    #     # trainSampler.set_epoch(self._epoch)

    #     # self.saver.debug("[%s] Epoch %4d started.", self.PrettyStep, self._epoch + 1)

    #     gc.collect()
    #     gc.collect()
    #     hook(self._step, self, *args, logger=self.saver, **kwArgs)


    def train(self, trainLoader: DataLoader, *_, beforeRunHook: Optional[Callable] = None, afterRunHook: Optional[Callable] = None, stepStartHook: Optional[Callable] = None, stepFinishHook: Optional[Callable] = None, **__):
        beforeRunHook = checkHook(beforeRunHook, "BeforeRunHook", self.saver)
        afterRunHook = checkHook(afterRunHook, "AfterRunHook", self.saver)
        stepStartHook = checkHook(stepStartHook, "StepStartHook", self.saver)
        stepFinishHook = checkHook(ChainHook(FrequecyHook((1000, self.periodicSave), logger=self.saver), stepFinishHook), "StepFinishHook", self.saver)
        # epochStartHook = checkHook(epochStartHook, "EpochStartHook", self.saver)
        # epochFinishHook = checkHook(epochFinishHook, "EpochFinishHook", self.saver)

        trainingArgs = {
            "trainLoader": trainLoader
        }

        self._beforeRun(beforeRunHook, **trainingArgs)

        scaler = GradScaler()

        images, xHat, codes = None, None, None

        localRank = int(os.environ['LOCAL_RANK'])

        while True:
            # self._epochStart(epochStartHook, **trainingArgs)
            for images in trainLoader:
                self.saver.debug("[%s] Image loaded.", self.PrettyStep)
                images = self.transform(images.to(localRank, non_blocking=True))

                self._stepStart(stepStartHook, **trainingArgs)

                self._optimizer.zero_grad()

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predictions, codes, xHat = self._model(images)
                self.saver.debug("[%s] Model forwarded.", self.PrettyStep)
                loss = sum([F.cross_entropy(pre, gt) for (pre, gt) in zip(predictions, codes[1:])])
                scaler.scale(loss).backward()
                # loss.backward()

                scaler.unscale_(self._optimizer)

                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 4.0)

                # self._optimizer.step()
                scaler.step(self._optimizer)
                self.saver.debug("[%s] Model backwarded.", self.PrettyStep)

                scaler.update()

                self._stepFinish(stepFinishHook, loss=loss, codes=codes, images=images, restored=xHat, **trainingArgs)
                del images
                if self._step >= self._totalStep:
                    break
                if self._step % 1000 == 0:
                    gc.collect()
                    gc.collect()

            if self._step >= self._totalStep:
                break
            # self._epochFinish(epochFinishHook, images=images, restored=xHat, codes=codes, logits=logits, **trainingArgs)
        self._afterRun(afterRunHook)


class MainGenTrainer(_baseGenTrainer):
    def __init__(self, config: Config, tmpFile: Optional[StrPath], modelFn: Callable[[], Tuple[BaseCompressor, Distortion]], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], saver: Saver):
        # global rank
        if dist.get_rank() != 0:
            raise AttributeError("A sub-process should not to be a <MainTrainer>, use <PalTrainer> instead.")

        self.rank = dist.get_rank()
        self.config = config
        self.saver = saver

        if self.rank == 0:
            wandb.login(key=os.environ['MCQUIC_WANDB_LOGIN'])
            self.run = wandb.init(
                project='mcquic_gen',
                config={
                    'model': config.Model.Params,
                    'batch_size': config.Train.BatchSize,
                    'total_step': config.Train.TotalStep,
                    # 'epoch': config.Train.Epoch,
                    'target': config.Train.Target,
                    'optim': {
                        'key': config.Train.Optim.Key,
                        'params': config.Train.Optim.Params
                    },
                    'schdr': {
                        'key': config.Train.Schdr.Key,
                        'params': config.Train.Schdr.Params
                    },
                    'hook': [
                        {'key': h.Key, 'params': h.Params} for h in config.Train.Hooks
                    ]
                },
                save_code=True,
                job_type='train',
                name=datetime.datetime.now().strftime(r'%m%d-%H:%M'),
            )

        self.progress = getRichProgress().__enter__()
        self.trainingBar = self.progress.add_task("", start=False, progress="[----/----]", suffix=Consts.CDot * 10)
        # self.epochBar = self.progress.add_task("[----/----]", start=False, progress="", suffix=Consts.CDot * 10)

        self.validator = Validator(self.config, int(os.environ['LOCAL_RANK']))

        self.diffTracker = EMATracker((), 0.99).to(int(os.environ['LOCAL_RANK']))

        # Call function at every X epoches.
        self.stepFinishCalls = FrequecyHook(
            (1000, self.log),
            logger=self.saver
        )
        self.stepStartCalls = FrequecyHook(
            (10000, self.validate),
            logger=self.saver
        )

        # self.bestRate = 1e10
        # self.bestDistortion = -1

        super().__init__(config, tmpFile, modelFn, optimizer, scheduler, saver)

        # signal.signal(signal.SIGTERM, self._terminatedHandler)

    # def _kill(self):
    #     sleep(Consts.TimeOut)
    #     self.saver.critical("[%s] Timeout exceeds, killed.", self.PrettyStep)
    #     signal.raise_signal(signal.SIGKILL)

    # # Handle SIGTERM when main process is terminated.
    # # Save necessary info.
    # def _terminatedHandler(self, signum, frame):
    #     killer = threading.Thread(target=self._kill, daemon=True)
    #     killer.start()
    #     self.saver.critical("[%s] Main process was interrupted, try to save necessary info.", self.PrettyStep)
    #     self.saver.critical("[%s] This post-process will be killed after %d secs if stuck.", self.PrettyStep, Consts.TimeOut)
    #     self.progress.__exit__(None, None, None)
    #     self.save(os.path.join(self.saver.SaveDir, "last.ckpt"))
    #     self.saver.critical("[%s] Find the last checkpoint at `%s`", self.PrettyStep, relativePath(os.path.join(self.saver.SaveDir, "last.ckpt")))
    #     self.summary()
    #     self.saver.critical("[%s] QUIT.", self.PrettyStep)
    #     # reset to default SIGTERM handler
    #     signal.signal(signal.SIGTERM, signal.SIG_DFL)
    #     signal.raise_signal(signal.SIGTERM)

    def summary(self):
        if abs(self.bestRate / self.bestDistortion) > 1e4:
            self.saver.info("[%s] Total steps: %s, loss: N/A.", self.PrettyStep, self.PrettyStep)
        else:
            self.saver.info("[%s] Total steps: %s, loss: %.2f.", self.PrettyStep, self.PrettyStep, 0.0)
        self.saver.info("[%s] Test this model by `python -m mcquic.validate --path %s`.", self.PrettyStep, relativePath(os.path.join(self.saver.SaveDir, "[ONE_OF_A].ckpt")))

    def train(self, trainLoader: DataLoader, valLoader: DataLoader, *_, beforeRunHook: Optional[Callable] = None, afterRunHook: Optional[Callable] = None, stepStartHook: Optional[Callable] = None, stepFinishHook: Optional[Callable] = None, **__):
        return super().train(trainLoader,
            beforeRunHook=beforeRunHook,
            afterRunHook=ChainHook(
                functools.partial(self.validate, valLoader=valLoader),
                afterRunHook),
            stepStartHook=ChainHook(
                functools.partial(self.stepStartCalls, valLoader=valLoader),
                stepStartHook),
            stepFinishHook=ChainHook(self._stepFinishHook, self.stepFinishCalls, stepFinishHook))

    def _beforeRun(self, hook, *args, **kwargs):
        self.progress.start_task(self.trainingBar)

        self.progress.update(self.trainingBar, total=self._totalStep)
        self.progress.reset(self.trainingBar)

        super()._beforeRun(hook, *args, **kwargs)
        # self.saver.info("[%s] See you at `%s`", self.PrettyStep, self.saver.TensorboardURL)

    def _afterRun(self, hook, *args, **kwArgs):
        self.progress.__exit__(None, None, None)
        super()._afterRun(hook, *args, **kwArgs)
        self.save(os.path.join(self.saver.SaveDir, "result.ckpt"))
        self.summary()

    def _stepFinishHook(self, *_, loss, **__):
        moment = self.diffTracker(loss)

        task = self.progress.get_task(self.trainingBar)
        self.progress.update(self.trainingBar, advance=1, progress=f"[{task.completed + 1:4d}/{task.total:4d}]", suffix=f"D = [b green]{moment:.2f}[/]")

        if self._step % 10 != 0:
            return
        if self.rank == 0:
            wandb.log({f"Stat/Loss": loss, "Stat/Lr": self._scheduler.get_last_lr()[0]}, step=self._step)
        if self._step % 100 == 0:
            if torch.isnan(moment):
                self.saver.critical('Loss becomes NAN. Train crashed.')
                raise RuntimeError('Loss becomes NAN. Train crashed.')
            self.saver.info('[%s / %s] Loss (CE): %.2f, Lr: %.1e, Est: %s', self.PrettyStep, self._formatStep(int(self._totalStep)), moment, self._scheduler.get_last_lr()[0], datetime.timedelta(seconds=self.progress.get_task(self.trainingBar).time_remaining))
        # self.saver.add_scalar(f"Stat/{self.config.Train.Target}", distortionDB, global_step=self._step)
        # self.saver.add_scalar(f"Stat/Rate", rate, global_step=self._step)
        # self.saver.add_scalar("Stat/Lr", self._scheduler.get_last_lr()[0], global_step=self._step)

    # def _epochStart(self, hook, *args, trainLoader: DataLoader, **kwArgs):
    #     import json
    #     def parseLengthFromMetadata(path):
    #         metadata = os.path.join(path, 'metadata.json')
    #         with open(metadata, 'r') as fp:
    #             metadata = json.load(fp)
    #         return metadata['length']

    #     # length = parseLengthFromMetadata(self.config.Train.TrainSet)
    #     # totalBatches = length // (self.worldSize * self.config.Train.BatchSize)
    #     self.progress.update(self.trainingBar, total=self._totalStep)
    #     # self.progress.update(self.epochBar, total=self.config.Train.Epoch * totalBatches, completed=self._step, description=f"[{self._epoch + 1:4d}/{self.config.Train.Epoch:4d}]")

    #     self.progress.reset(self.trainingBar)

    #     # super()._epochStart(hook, *args, **kwArgs)


    def log(self, *_, images, restored, codes, **__):
        if self.rank != 0:
            return
        payload: Dict[str, Any] = dict()
        # self.saver.add_scalar("Stat/Epoch", self._epoch, self._step)
        # First level, first image, first group
        # self.saver.add_histogram("Stat/LogDistance", (-(logits[0][0, 0])).clamp(Consts.Eps).log10(), global_step=self._step)
        # [m, ki]
        for lv, c in enumerate(codes):
            payload[f'Hist/CodeLv{lv}'] = [wandb.Image(to_pil_image(x)) for x in self.validator.visualizeIntermediate(c)]
            # self.saver.add_histogram_raw(f"Stat/FreqLv{lv}", min=0, max=len(fr[0]), num=len(fr[0]), sum=fr[0].sum(), sum_squares=(fr[0] ** 2).sum(), bucket_limits=list(range(len(fr[0]))), bucket_counts=fr[0], global_step=self._step)
            # self.saver.add_images(f"Train/CodeLv{lv}", self.validator.visualizeIntermediate(c), self._step)
        payload['Train/Raw'] = [wandb.Image(to_pil_image(x)) for x in self.validator.tensorToImage(images)]
        # self.saver.add_images("Train/Raw", self.validator.tensorToImage(images), global_step=self._step)
        # self.saver.add_images("Train/Post", self.validator.tensorToImage(postProcessed), global_step=self._step)

        payload['Train/Res'] = [wandb.Image(to_pil_image(x)) for x in self.validator.tensorToImage(restored)]
        # self.saver.add_images("Train/Res", self.validator.tensorToImage(restored), global_step=self._step)

        # self.saver.add_scalar("Stat/CodeUsage", self._model.Compressor.CodeUsage, global_step=self._step)

        wandb.log(payload, step=self._step)

        self.saver.debug('[%s] `MainTrainaer.log` finished.', self.prettyStep)

    def validate(self, *_, valLoader: DataLoader, **__):
        return
        torch.cuda.empty_cache()

        self.saver.debug("[%s] Start validation.", self.PrettyStep)

        self._model.eval()
        results, summary = self.validator.validate(0, self._model.Compressor, valLoader, self.progress)


        wandb.log({
            'Eval/MsSSIM': results["MsSSIM"],
            'Eval/PSNR': results["PSNR"],
            'Eval/BPP': results['BPP'],
            'Eval/Visualization': [wandb.Image(to_pil_image(x)) for x in results["Visualization"]]
        }, step=self._step)

        # self.saver.add_scalar(f"Eval/MsSSIM", results["MsSSIM"], global_step=self._step)
        # self.saver.add_scalar(f"Eval/PSNR", results["PSNR"], global_step=self._step)
        # self.saver.add_scalar(f"Eval/BPP", results["BPP"], global_step=self._step)
        # self.saver.add_images(f"Eval/Visualization", results["Visualization"], global_step=self._step)

        rate, distortion = results["BPP"], results[self.config.Train.Target]

        self.save(os.path.join(self.saver.SaveDir, f"val_{self._step}.ckpt"))

        # TODO: Why d/r continously decrease?
        # if (distortion / rate) > (self.bestDistortion / self.bestRate):
        if distortion > self.bestDistortion:
            self.bestDistortion = distortion
            self.bestRate = rate
            # self.progress.update(self.epochBar, suffix=f"H = [b red]{self.bestDistortion:2.2f}[/]dB")
            shutil.copy2(os.path.join(self.saver.SaveDir, f"val_{self._step}.ckpt"), os.path.join(self.saver.SaveDir, "best.ckpt"))
        self.saver.info("[%s] %s", self.PrettyStep, summary)
        self._model.train()

        self.saver.debug("[%s] End validation.", self.PrettyStep)


class PalGenTrainer(_baseGenTrainer):
    def __init__(self, config: Config, tmpFile: Optional[StrPath], modelFn: Callable[[], Tuple[BaseCompressor, Distortion]], optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], saver: Saver):
        if dist.get_rank() == 0:
            raise AttributeError("You should call <MainTrainer> for main process other than <PalTrainer> to save, log necessary information.")
        super().__init__(config, tmpFile, modelFn, optimizer, scheduler, saver)

    def train(self, trainLoader: DataLoader, *_, beforeRunHook: Optional[Callable] = None, afterRunHook: Optional[Callable] = None, stepStartHook: Optional[Callable] = None, stepFinishHook: Optional[Callable] = None, **__):
        return super().train(trainLoader, beforeRunHook=beforeRunHook, afterRunHook=afterRunHook, stepStartHook=stepStartHook, stepFinishHook=stepFinishHook)
