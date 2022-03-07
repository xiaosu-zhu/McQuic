import math
from typing import Any, Dict

from marshmallow import Schema, fields, post_load


class GeneralSchema(Schema):
    type = fields.Str()
    params = fields.Dict(keys=fields.Str())

    @post_load
    def _(self, data, **kwargs):
        return General(**data)

class TrainingSchema(Schema):
    batchSize = fields.Int()
    epoch = fields.Int()
    valFreq = fields.Int()
    trainSet = fields.Str()
    valSet = fields.Str()
    saveDir = fields.Str()
    target = fields.Str()

    @post_load
    def _(self, data, **kwargs):
        return Training(**data)

class GPUSchema(Schema):
    gpus = fields.Int()
    vRam = fields.Int()
    wantsMore = fields.Bool()

    @post_load
    def _(self, data, **kwargs):
        return GPU(**data)

class ConfigSchema(Schema):
    model = fields.Nested(GeneralSchema())
    optim = fields.Nested(GeneralSchema())
    schdr = fields.Nested(GeneralSchema())
    training = fields.Nested(TrainingSchema())
    gpu = fields.Nested(GPUSchema())

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


class Training:
    def __init__(self, batchSize: int, epoch: int, valFreq: int, trainSet: str, valSet: str, saveDir: str, target: str):
        self.batchSize = batchSize
        self.epoch = epoch
        self.valFreq = valFreq
        self.trainSet = trainSet
        self.valSet = valSet
        self.saveDir = saveDir
        self.target = target

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


class GPU:
    def __init__(self, gpus: int, vRam: int, wantsMore: bool) -> None:
        self.gpus = gpus
        self.vRam = vRam
        self.wantsMore = wantsMore

    @property
    def GPUs(self) -> int:
        return self.gpus

    @property
    def VRam(self) -> int:
        return self.vRam

    @property
    def WantsMore(self) -> bool:
        return self.wantsMore


class Config:
    def __init__(self, model: General, optim: General, schdr: General, training: Training, gpu: GPU):
        self.model = model
        self.optim = optim
        self.schdr = schdr
        self.training = training
        self.gpu = gpu

    @property
    def Model(self) -> General:
        return self.model

    @property
    def Optim(self) -> General:
        return self.optim

    @property
    def Schdr(self) -> General:
        return self.schdr

    @property
    def Training(self) -> Training:
        return self.training

    @property
    def GPU(self) -> GPU:
        return self.gpu

    def scaleByWorldSize(self, worldSize: int):
        batchSize = self.training.BatchSize * worldSize
        exponent = math.log2(batchSize)
        scale = 3 - exponent / 2
        if "lr" in self.Optim.params:
            self.Optim.params["lr"] /= (2 ** scale)

    def serialize(self) -> dict:
        return ConfigSchema().dump(self) # type: ignore

    @staticmethod
    def deserialize(data: dict) -> "Config":
        return ConfigSchema().load(data) # type: ignore
