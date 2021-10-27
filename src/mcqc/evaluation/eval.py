from collections import ChainMap
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ConvertImageDtype
from cfmUtils.config import read

from mcqc.evaluation.refModel import PostProcess, Preprocess
from mcqc.evaluation.tests import Performance, Speed, Test
from absl import app
from absl import flags
from mcqc import Config
from mcqc.datasets import Basic

FLAGS = flags.FLAGS

flags.DEFINE_string("cfg", "", "The config.json path.")
flags.DEFINE_string("device", "cuda", "The device to use.")
flags.DEFINE_string("encoder", "", "The encoder.ckpt file path.")
flags.DEFINE_string("decoder", "", "The decoder.ckpt file path.")
flags.DEFINE_string("dataset", "data/clic/valid", "The images path")

class Eval:
    def __init__(self, device: str, encoderPath: str, decoderPath: str, config: Config, dataset: Dataset):
        self._encoder = torch.jit.load(encoderPath, map_location=device)
        self._decoder = torch.jit.load(decoderPath, map_location=device)
        self._preProcess = torch.jit.script(Preprocess(128)).to(device)
        self._postProcess = torch.jit.script(PostProcess()).to(device)
        self._dataset = dataset

        generalArgs = {
            "device": device,
            "encoder": self._encoder,
            "decoder": self._decoder,
            "config": config,
            "preProcess": self._preProcess,
            "postProcess": self._postProcess
        }

        self._tests: List[Test] = [#Performance(dataset, **generalArgs),
         Speed(**generalArgs)]

    def __call__(self):
        results = dict(ChainMap(*[x.test() for x in self._tests]))
        print(results)


@torch.inference_mode()
def main(_):
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    config = read(FLAGS.cfg, None, Config)
    Eval(FLAGS.device, FLAGS.encoder, FLAGS.decoder, config, Basic(FLAGS.dataset, transform=ConvertImageDtype(torch.float32)))()


if __name__ == "__main__":
    app.run(main)
