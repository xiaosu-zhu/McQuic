"""Package of algorithms

Exports:
    ProximalPolicy: Original Proximal Policy Optimization algorithm.
    VCProximalPolicy: Value-Corrected Proximal Policy Optimization algorithm.
    SoftActorCritic: Soft Actor-Critic algorithm.
"""
from .plain import Plain
from .fineTune import FineTune
from .twoPass import TwoPass
from .new import New
