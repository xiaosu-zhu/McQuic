from dataclasses import dataclass


@dataclass
class Config:
    batchSize: int = 32
    epoch: int = 10
    gpus: int = 1
    vRam: int = 8000
    wantsMore: bool = False
    dataset: str = "clic/train"
    valDataset: str = "clic/valid"

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
