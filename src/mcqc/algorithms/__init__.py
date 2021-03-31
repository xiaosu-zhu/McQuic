"""Package of algorithms

Exports:
    ProximalPolicy: Original Proximal Policy Optimization algorithm.
    VCProximalPolicy: Value-Corrected Proximal Policy Optimization algorithm.
    SoftActorCritic: Soft Actor-Critic algorithm.
"""
from .plain import Plain
from .gan import TwoStageWithGan
from .reinforce import Reinforce
from .storch import Storch
from .twoStage import TwoStage
from .explicitTwo import ExpTwoStage
