
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from vlutils.metrics.meter import Meters
from rich.progress import Progress

from mcqc.utils.vision import DeTransform
from mcqc.validation.meters import MsSSIM, PSNR, BPP
from mcqc.models.compressor import BaseCompressor


class Validator:
    def __init__(self, rank: int):
        self.rank = rank
        self._deTrans = DeTransform().to(rank)
        self._meter = Meters(handlers=[
            MsSSIM().to(rank),
            PSNR().to(rank),
            BPP().to(rank)
        ])

    def tensorToImage(self, x: torch.Tensor) -> torch.Tensor:
        return self._deTrans(x)

    def visualizeIntermediate(self, code: torch.Tensor) -> torch.Tensor:
        code = self._deTrans((code.float() / code.max() - 0.5) * 2)

        n, m, h, w = code.shape

        code = code.reshape(n * m, 1, h, w)[:32]
        code = F.interpolate(code, scale_factor=4, mode="nearest")
        return code

    @torch.inference_mode()
    def count(self, epoch: int, model: BaseCompressor, trainLoader: DataLoader, progress: Progress):
        total = len(trainLoader)
        now = 0
        task = progress.add_task(f"Stat@{epoch:4d}", total=total, progress=f"{now:4d}/{total:4d}", suffix="")
        for now, image in enumerate(trainLoader):
            model.count(image.to(self.rank, non_blocking=True))
            progress.update(task, advance=1, progress=f"{(now + 1):4d}/{total:4d}")
        progress.remove_task(task)

    @torch.inference_mode()
    def validate(self, epoch: int, model: BaseCompressor, valLoader: DataLoader, progress: Progress):
        total = len(valLoader)
        now = 0
        task = progress.add_task(f"Val@{epoch:4d}", total=total, progress=f"{now:4d}/{total:4d}", suffix="")
        with model._quantizer.readyForCoding() as cdfs:
            for now, images in enumerate(valLoader):
                images = images.to(self.rank, non_blocking=True)
                codes, binaries, header = model.compress(images, cdfs)
                restored = model.decompress(codes, binaries, cdfs, header)
                self._meter(images=self.tensorToImage(images), binaries=binaries, restored=self.tensorToImage(restored))
                progress.update(task, advance=1, progress=f"{(now + 1):4d}/{total:4d}")
        progress.remove_task(task)
        return self._meter.results(), self._meter.summary()

    def test(self, testLoader: DataLoader): raise NotImplementedError
