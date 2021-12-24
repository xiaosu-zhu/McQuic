
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from mcqc.utils.transforms import DeTransform
from mcqc.evaluation.metrics import MsSSIM, PSNR


class Validator:
    def __init__(self, rank: int):
        self._valSSIM = MsSSIM(sizeAverage=False).to(rank)
        self._valPSNR = PSNR(sizeAverage=False).to(rank)
        self._deTrans = DeTransform().to(rank)

    def tensorToImage(self, x: torch.Tensor) -> torch.Tensor:
        return self._deTrans(x)

    def visualizeIntermediate(self, code: torch.Tensor) -> torch.Tensor:
        code = self._deTrans((code.float() / code.max() - 0.5) * 2)

        n, m, h, w = code.shape

        code = code.reshape(n * m, 1, h, w)[:32]
        code = F.interpolate(code, scale_factor=4, mode="nearest")
        return code

    def validate(self, valLoader: DataLoader):
        raise NotImplementedError

    def test(self, testLoader: DataLoader): raise NotImplementedError
