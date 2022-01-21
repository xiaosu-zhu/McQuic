from collections import ChainMap
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ConvertImageDtype
from vlutils.config import read

from mcquic.evaluation.refModel import PostProcess, Preprocess
from mcquic.evaluation.tests import Performance, Speed, Test, Preparar
from absl import app
from absl import flags
from mcquic import Config
from mcquic.datasets import Basic

FLAGS = flags.FLAGS

flags.DEFINE_string("cfg", "", "The config.json path.")
flags.DEFINE_string("device", "cuda", "The device to use.")
flags.DEFINE_string("encoder", "", "The encoder.ckpt file path.")
flags.DEFINE_string("decoder", "", "The decoder.ckpt file path.")
flags.DEFINE_string("dataset", "data/clic/valid", "The images path")

class Eval:
    def __init__(self, device: str, encoderPath: str, decoderPath: str, config: Config, dataset: Dataset):
        self._encoder = torch.jit.load(encoderPath, map_location=device).eval()
        self._decoder = torch.jit.load(decoderPath, map_location=device).eval()
        self._preProcess = torch.jit.script(Preprocess(128)).to(device).eval()
        self._postProcess = torch.jit.script(PostProcess()).to(device).eval()
        self._dataset = dataset

        generalArgs = {
            "device": device,
            "encoder": self._encoder,
            "decoder": self._decoder,
            "config": config,
            "preProcess": self._preProcess,
            "postProcess": self._postProcess
        }

        self._preparar = Preparar(dataset, **generalArgs)

        self._tests: List[Test] = [Performance(dataset, **generalArgs)] # [Speed(**generalArgs)] # [Performance(dataset, **generalArgs),Speed(**generalArgs)]

    def __call__(self):
        cdfs = self._preparar.test()
        results = dict(ChainMap(*[x.test(cdfs) for x in self._tests]))
        print(results)


@torch.inference_mode()
def main(_):
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    config = read(FLAGS.cfg, None, Config)
    Eval(FLAGS.device, FLAGS.encoder, FLAGS.decoder, config, Basic(FLAGS.dataset, transform=ConvertImageDtype(torch.float32)))()


if __name__ == "__main__":
    app.run(main)
