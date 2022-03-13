from dataclasses import dataclass
import math
from typing import Any, Dict

from marshmallow import Schema, fields, post_load


class GeneralSchema(Schema):
    type = fields.Str(description="A unique key used to retrieve in registry. For example, given `Lamb` for optimizers, it will check `OptimRegistry` and find the optimizer `apex.optim.FusedLAMB`.")
    params = fields.Dict(keys=fields.Str(), description="Corresponding funcation call parameters.")

    @post_load
    def _(self, data, **kwargs):
        return General(**data)

class GPUSchema(Schema):
    gpus = fields.Int(description="Number of gpus for training. This affects the `world size` of PyTorch DDP.", exclusiveMinimum=0)
    vRam = fields.Int(description="Minimum VRam required for each gpu. Set it to `-1` to use all gpus.")
    wantsMore = fields.Bool(description="Set to `true` to use all visible gpus and all VRams and ignore `gpus` and `vRam`.")

    @post_load
    def _(self, data, **kwargs):
        return GPU(**data)

class TrainSchema(Schema):
    batchSize = fields.Int(description="Batch size for training. NOTE: The actual batch size (whole world) is computed by `batchSize * gpus`.", exclusiveMinimum=0)
    epoch = fields.Int(description="Total training epochs.", exclusiveMinimum=0)
    valFreq = fields.Int(description="Run validation after every `valFreq` epochs.", exclusiveMinimum=0)
    trainSet = fields.Str(description="A dir path to load `lmdb` dataset. You need to convert your images before you give this path by calling `mcquic dataset ...`.")
    valSet = fields.Str(description="A dir path to load image files for validation.")
    saveDir = fields.Str(description="A dir path to save model checkpoints, TensorBoard messages and logs.")
    target = fields.Str(description="Training target. Now is one of `[PSNR, MsSSIM]`.", enum=["PSNR", "MsSSIM"])
    optim = fields.Nested(GeneralSchema(), description="Optimizer used for training. Now we have `Adam` and `Lamb`.")
    schdr = fields.Nested(GeneralSchema(), description="Learning rate scheduler used for training. Now we have `ReduceLROnPlateau`, `Exponential`, `MultiStep`, `OneCycle` and all schedulers defined in `mcquic.train.lrSchedulers`.")
    gpu = fields.Nested(GPUSchema(), description="GPU configs for training.")

    @post_load
    def _(self, data, **kwargs):
        return Train(**data)

class ConfigSchema(Schema):
    model = fields.Nested(GeneralSchema(), description="Compression model to use. Now we only have one model, so `type` is ignored. Avaliable params are `channel`, `m` and `k`.")
    train = fields.Nested(TrainSchema(), description="Training configs.")

    @post_load
    def _(self, data, **kwargs):
        return Config(**data)


class General:
    def __init__(self, type: str, params: Dict[str, Any]):
        self.type = type
        self.params = params

    @property
    def Type(self) -> str:
        return self.type

    @property
    def Params(self) -> Dict[str, Any]:
        return self.params



@dataclass
class GPU:
    gpus: int
    vRam: int
    wantsMore: bool

    @property
    def GPUs(self) -> int:
        return self.gpus

    @property
    def VRam(self) -> int:
        return self.vRam

    @property
    def WantsMore(self) -> bool:
        return self.wantsMore


@dataclass
class Train:
    batchSize: int
    epoch: int
    valFreq: int
    trainSet: str
    valSet: str
    saveDir: str
    target: str
    optim: General
    schdr: General
    gpu: GPU

    @property
    def BatchSize(self) -> int:
        return self.batchSize

    @property
    def Epoch(self) -> int:
        return self.epoch

    @property
    def ValFreq(self) -> int:
        return self.valFreq

    @property
    def TrainSet(self) -> str:
        return self.trainSet

    @property
    def ValSet(self) -> str:
        return self.valSet

    @property
    def SaveDir(self) -> str:
        return self.saveDir

    @property
    def Target(self) -> str:
        return self.target

    @property
    def Optim(self) -> General:
        return self.optim

    @property
    def Schdr(self) -> General:
        return self.schdr

    @property
    def GPU(self) -> GPU:
        return self.gpu


@dataclass
class Config:
    model: General
    train: Train

    @property
    def Model(self) -> General:
        return self.model

    @property
    def Train(self) -> Train:
        return self.train

    def scaleByWorldSize(self, worldSize: int):
        batchSize = self.train.BatchSize * worldSize
        exponent = math.log2(batchSize)
        scale = 3 - exponent / 2
        if "lr" in self.Train.Optim.params:
            self.Train.Optim.params["lr"] /= (2 ** scale)

    def serialize(self) -> dict:
        return ConfigSchema().dump(self) # type: ignore

    @staticmethod
    def deserialize(data: dict) -> "Config":
        return ConfigSchema().load(data) # type: ignore
# let's try to trigger the action
