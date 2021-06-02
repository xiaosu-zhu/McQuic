from typing import List, Union
from dataclasses import dataclass


@dataclass
class Coef:
    ssim: float = 2.0
    l1l2: float = 2.0
    reg: float = 0.2
    gen: float = 0.1
    dis: float = 0.1


@dataclass
class ModelSpec:
    type: str
    k: List[int]
    channel: int = 512
    numLayers: int = 3


@dataclass
class Config:
    lr: float = 5e-6
    coef: Coef = Coef()
    model: ModelSpec = ModelSpec(type="Plain", k=[256])
    batchSize: int = 32
    epoch: int = 10
    gpus: int = 1
    vRam: int = 8000
    wantsMore: bool = False
    dataset: str = "clic/train"
    valDataset: str = "clic/valid"
    method: str = "Plain"

    @property
    def LearningRate(self) -> float:
        return self.lr

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
