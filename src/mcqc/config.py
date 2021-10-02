from typing import List
from dataclasses import dataclass


@dataclass
class Coef:
    ssim: float = 2.0
    l1l2: float = 2.0


@dataclass
class ModelSpec:
    type: str
    k: List[int]
    target: str
    m: int = 4
    withGroup: bool = True
    channel: int = 256
    withAtt: bool = True
    alias: bool = True
    ema: float = 0.8


@dataclass
class OptimSpec:
    type: str
    params: dict


@dataclass
class SchdrSpec:
    type: str
    params: dict


@dataclass
class RegSchdrSpec:
    type: str
    params: dict


@dataclass
class Config:
    coef: Coef = Coef()
    model: ModelSpec = ModelSpec(type="Base", target="ssim", m=8, k=[2048, 512, 128])
    optim: OptimSpec = OptimSpec(type="Adam", params={})
    schdr: SchdrSpec = SchdrSpec(type="ReduceLROnPlateau", params={})
    regSchdr: RegSchdrSpec = RegSchdrSpec(type="Step", params={})
    batchSize: int = 4
    epoch: int = 10
    gpus: int = 1
    vRam: int = -1
    wantsMore: bool = False
    dataset: str = "clic/train"
    valDataset: str = "clic/valid"
    method: str = "Plain"
    evalStep: int = 1000
    testStep: int = 10000
    warmStart: str = "ckpt/global.ckpt"
    repeat: int = 1

    @property
    def Repeat(self) -> int:
        return self.repeat

    @property
    def Optim(self) -> OptimSpec:
        return self.optim

    @property
    def Schdr(self) -> SchdrSpec:
        return self.schdr

    @property
    def RegSchdr(self) -> RegSchdrSpec:
        return self.regSchdr

    @property
    def WarmStart(self) -> str:
        return self.warmStart

    @property
    def EvalStep(self) -> int:
        return self.evalStep

    @property
    def TestStep(self) -> int:
        return self.testStep

    @property
    def Model(self) -> ModelSpec:
        return self.model

    @property
    def Coef(self) -> Coef:
        return self.coef

    @property
    def BatchSize(self) -> int:
        return self.batchSize

    @property
    def Epoch(self) -> int:
        return self.epoch

    @property
    def GPUs(self) -> int:
        return self.gpus

    @property
    def VRam(self) -> int:
        return self.vRam

    @property
    def WantsMore(self) -> bool:
        return self.wantsMore

    @property
    def Dataset(self) -> str:
        return self.dataset

    @property
    def ValDataset(self) -> str:
        return self.valDataset

    @property
    def Method(self) -> str:
        return self.method
