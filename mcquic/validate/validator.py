from typing import Optional
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from vlutils.metrics.meter import Meters
from rich.progress import Progress
from mcquic.config import Config

from mcquic.utils.vision import DeTransform
from mcquic.validate.handlers import MsSSIM, PSNR, BPP, IdealBPP, Visualization, ImageCollector
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
            IdealBPP([1] * len(config.Model.Params['size']), [config.Model.Params['k']] * len(config.Model.Params['size'])).to(rank),
            ImageCollector().to(rank)
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
    def validate(self, epoch: Optional[int], model: BaseCompressor, valLoader: DataLoader, progress: Progress):
        self._meter.reset()
        total = len(valLoader)
        now = 0
        if epoch is None:
            # test mode
            task = progress.add_task(f"[ Test ]", total=total, progress=f"{now:4d}/{total:4d}", suffix="")
        else:
            task = progress.add_task(f"[ Val@{epoch:4d}]", total=total, progress=f"{now:4d}/{total:4d}", suffix="")

        for now, (images, stem) in enumerate(valLoader):
            images = images.to(self._rank, non_blocking=True)
            codes, binaries, headers = model.compress(images)
            restored = model.decompress(binaries, headers)
            self._meter(images=self.tensorToImage(images), binaries=binaries, restored=self.tensorToImage(restored), codes=codes, stem=stem)
            progress.update(task, advance=1, progress=f"{(now + 1):4d}/{total:4d}")
        progress.remove_task(task)
        return self._meter.results(), self._meter.summary()

    @torch.inference_mode()
    def speed(self, epoch: int, model: BaseCompressor, progress: Progress):
        now = 0
        if epoch is None:
            # test mode
            task = progress.add_task(f"[ Speed test ]", total=100, progress=f"{now:4d}/{100:4d}", suffix="")
        else:
            task = progress.add_task(f"[ Spd@{epoch:4d}]", total=100, progress=f"{now:4d}/{100:4d}", suffix="")

        tensor = torch.rand(10, 3, 768, 512).to(self._rank)

        startEvent = torch.cuda.Event(enable_timing=True)
        endEvent = torch.cuda.Event(enable_timing=True)

        # warm up
        codes, binaries, headers = model.compress(tensor)
        restored = model.decompress(binaries, headers)

        startEvent.record()
        for _ in range(50):
            codes, binaries, headers = model.compress(tensor)
            progress.update(task, advance=1, progress=f"{(now + 1):4d}/{100:4d}")
            now += 1
        endEvent.record()
        torch.cuda.synchronize()
        encoderMs = startEvent.elapsed_time(endEvent)

        startEvent.record()
        for _ in range(50):
            restored = model.decompress(binaries, headers)
            progress.update(task, advance=1, progress=f"{(now + 1):4d}/{100:4d}")
            now += 1
        endEvent.record()
        torch.cuda.synchronize()
        decoderMs = startEvent.elapsed_time(endEvent)

        result = ((50 * 10 * 768 * 512 / 1000) / encoderMs, (50 * 10 * 768 * 512 / 1000) / decoderMs)
        return result, f"Coding throughput: encoder: {result[0]:.2f} Mpps, decoder: {result[1]:.2f} Mpps"
