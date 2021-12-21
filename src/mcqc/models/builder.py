
from collections import OrderedDict

from torch import nn


def build(registry, moduleSpec: OrderedDict):
    layers = list()
    for key, (args, kwArgs) in moduleSpec.items():
        layers.append(
            registry.get(key)(*args, **kwArgs)
        )
    return nn.Sequential(*layers)
