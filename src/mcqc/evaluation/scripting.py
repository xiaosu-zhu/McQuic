import os
from typing import Dict

from cfmUtils.config import read
from mcqc import Config

from absl import app
from absl import flags

import torch
from torch import nn

from mcqc.evaluation.refModel import PostProcess, Preprocess, RefEncoder, RefDecoder

FLAGS = flags.FLAGS

flags.DEFINE_string("cfg", None, "The config.json path.")
flags.DEFINE_string("path", None, "The .ckpt file path.")
flags.DEFINE_string("target", None, "The saving path.")
flags.DEFINE_boolean("gpu", True, "Whether the model should be on GPU.")


def migrate(model: nn.Module, prefix: str, stateDict: Dict[str, torch.Tensor]):
    stateDict = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in stateDict.items()}
    keys = model.keys
    stateDict = {k: v for k, v in stateDict.items() if k.startswith(keys)}
    model.load_state_dict(stateDict)
    return model


def convert(preProcess: nn.Module, encoder: nn.Module, decoder: nn.Module, postProcess: nn.Module, path: str):
    preProcess.eval()
    encoder.eval()
    decoder.eval()
    postProcess.eval()
    testInput = torch.rand([1, 3, 1357, 2468])
    try:
        testOutput = postProcess(*decoder(*encoder(*preProcess(testInput))))
        if testInput.shape != testOutput.shape:
            raise RuntimeError(f"Input-output shape mismatch, expected {testInput.shape}, got {testOutput.shape}, please check model.")
        if torch.any(testOutput < 0) or torch.any(testOutput > 1):
            raise RuntimeError("Find result out of range 0~1, please check model.")
    except Exception as e:
        raise RuntimeError(f"Model error with input {testInput.shape}.") from e
    testInput = torch.rand([1, 3, 2, 2])
    try:
        testOutput = postProcess(*decoder(*encoder(*preProcess(testInput))))
        if testInput.shape != testOutput.shape:
            raise RuntimeError(f"Input-output shape mismatch, expected {testInput.shape}, got {testOutput.shape}, please check model.")
        if torch.any(testOutput < 0) or torch.any(testOutput > 1):
            raise RuntimeError("Find result out of range 0~1, please check model.")
    except Exception as e:
        raise RuntimeError(f"Model error with input {testInput.shape}.") from e
    if FLAGS.gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    with torch.jit.optimized_execution(True):
        scriptedEncoder = torch.jit.script(encoder)
        scriptedEncoder.save(os.path.join(path, "encoder.pt"))
        del scriptedEncoder
        scriptedDecoder = torch.jit.script(decoder)
        scriptedDecoder.save(os.path.join(path, "decoder.pt"))


def _main(encoder, decoder, prefix, ckptPath, targetDir):
    torch.autograd.set_grad_enabled(False)
    if ckptPath is not None:
        stateDict = torch.load(ckptPath, map_location={"cuda:0": "cpu"})["model"]
        encoder = migrate(encoder, prefix, stateDict)
        decoder = migrate(decoder, prefix, stateDict)

    convert(Preprocess(128), encoder, decoder, PostProcess(), targetDir)

@torch.inference_mode()
def main(_):
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    config = read(FLAGS.cfg, None, Config)
    if not FLAGS.target or not os.path.isdir(FLAGS.target):
        raise FileNotFoundError(f"The given target: {FLAGS.target} is not a valid dir or doesn't exist.")
    if not FLAGS.path or not os.path.isfile(FLAGS.path):
        _main(RefEncoder(config.Model.m, config.Model.k, config.Model.channel, 1, config.Model.alias), RefDecoder(config.Model.m, config.Model.k, config.Model.channel, config.Model.alias), "module._compressor.", None, FLAGS.target)
    _main(RefEncoder(config.Model.m, config.Model.k, config.Model.channel, 1, config.Model.alias), RefDecoder(config.Model.m, config.Model.k, config.Model.channel, config.Model.alias), "module._compressor.", FLAGS.path, FLAGS.target)


if __name__ == "__main__":
    app.run(main)
