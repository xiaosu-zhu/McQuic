import threading

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from vlutils.metrics.meter import Meters
from rich.progress import Progress
from mcquic.config import Config

from mcquic.utils.vision import DeTransform
from mcquic.validate.handlers import MsSSIM, PSNR, BPP, IdealBPP, Visualization
from mcquic.modules.compressor import BaseCompressor


class Validator:
    def __init__(self, config: Config, rank: int):
        self._rank = rank
        self._deTrans = DeTransform().to(rank)
        self._meter = Meters(handlers=[
            MsSSIM().to(rank),
            PSNR().to(rank),
            BPP().to(rank),
            Visualization().to(rank),
            IdealBPP(config.Model.k).to(rank)
        ])

    def tensorToImage(self, x: torch.Tensor) -> torch.Tensor:
        return self._deTrans(x)

    def visualizeIntermediate(self, code: torch.Tensor) -> torch.Tensor:
        code = self._deTrans((code.float() / code.max() - 0.5) * 2)

        # n, m, h, w = code.shape
        # code = code.reshape(n * m, 1, h, w)[:32]

        # only visualize first group
        code = F.interpolate(code[:, :1], scale_factor=4, mode="nearest")
        return code

    @torch.inference_mode()
    def validate(self, epoch: int, model: BaseCompressor, valLoader: DataLoader, progress: Progress):
        self._meter.reset()
        total = len(valLoader)
        now = 0
        task = progress.add_task(f"[ Val@{epoch:4d}]", total=total, progress=f"{now:4d}/{total:4d}", suffix="")
        with model._quantizer.readyForCoding() as cdfs:
            for now, images in enumerate(valLoader):
                images = images.to(self._rank, non_blocking=True)
                codes, binaries, headers = model.compress(images, cdfs)
                restored = model.decompress(binaries, cdfs, headers)
                self._meter(images=self.tensorToImage(images), binaries=binaries, restored=self.tensorToImage(restored), codes=codes)
                progress.update(task, advance=1, progress=f"{(now + 1):4d}/{total:4d}")
        progress.remove_task(task)
        return self._meter.results(), self._meter.summary()

    def test(self, testLoader: DataLoader): raise NotImplementedError
