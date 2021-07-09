from collections import ChainMap
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ConvertImageDtype

from mcqc.config import ModelSpec
from mcqc.evaluation.refModel import RefDecoder, RefEncoder
from mcqc.evaluation.scripting import migrate
from mcqc.evaluation.tests import Performance, Speed, Test
from mcqc import Config
from mcqc.datasets import Basic


class Eval:
    def __init__(self, encoderPath: str, decoderPath: str, config: Config, device: int, dataset: Dataset):
        self._encoder = torch.jit.load(encoderPath)
        self._decoder = torch.jit.load(decoderPath)
        self._dataset = dataset

        self._tests: List[Test] = [Performance(dataset, config=config, encoder=self._encoder, decoder=self._decoder, device=device)] #, Speed(config=config, encoder=self._encoder, decoder=self._decoder, device=device)]

    def __call__(self):
        results = dict(ChainMap(*[x.test() for x in self._tests]))
        print(results)


if __name__ == "__main__":
    Eval("ckpt/encoder.pt", "ckpt/decoder.pt", Config(model=ModelSpec(type="Base", m=32, channel=256)), 0, Basic("data/clic/valid", transform=ConvertImageDtype(torch.float32)))()
