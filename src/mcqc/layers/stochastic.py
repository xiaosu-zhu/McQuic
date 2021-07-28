import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from mcqc.evaluation.metrics import MsSSIM
EPSILON = 1e-16


def topGumbel(gumbels, logits, samples):
    # [k, ?, d] + [?, d]
    # -log(Ej) + log(Z(θ))
    topGumbels = gumbels + logits.logsumexp(dim=-1, keepdim=True)
    # we only want value where j == i, i.e. the one-hot position
    # [k, ?, d]
    jiGumbels = samples * topGumbels
    return jiGumbels


def truncatedGumbel(gumbels, logits, samples, jiGumbels):
    # sum to get log(Z(θ) / Ei) for broadcasting
    topGumbel = jiGumbels.sum(-1, keepdim=True)
    # [k, ?, d]
    # θ - log(Ej)
    reGumbel = gumbels + logits
    # -log(exp(log(Ej) - θ) + exp(-log(Z(θ) / Ei)))
    # = -log(Ej/exp(θ) + Ei/Z(θ))
    truncated = -(EPSILON + (-reGumbel).exp() + (-topGumbel).exp()).log()
    # we only want value where j != i
    return (1 - samples) * truncated


def gumbelRaoMCK(logits, tau, k):
    samples = torch.distributions.OneHotCategorical(logits=logits).sample(())
    # [k, ?, d]
    # -log(Ej)
    gumbels = -torch.empty([k, *logits.shape], memory_format=torch.legacy_contiguous_format, device=logits.device).exponential_().log()
    jiGumbels = topGumbel(gumbels, logits, samples)
    truncated = truncatedGumbel(gumbels, logits, samples, jiGumbels)
    # Equation (9) in paper
    merged = truncated + jiGumbels
    # [?, d]
    temperedSoftmax = (merged / tau).softmax(-1).mean(0)
    return samples - temperedSoftmax.detach() + temperedSoftmax


# https://arxiv.org/pdf/1810.00116.pdf
# Improved Gradient-Based Optimization Over Discrete Distributions
# improved Gumbel-Softmax from Sec. 3.5
def iGumbelSoftmax(logits, tau, hard, dim=-1):
    logExp = logits.softmax(-1)
    gumbels = (
        -(torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_() - logExp + logExp.detach()).log()
    )  # ~Gumbel(0,1)
    # logExp = logits.log_softmax(-1)
    # gumbels = (
    #     -(torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_()).log() + logExp - logExp.detach()
    # )  # ~Gumbel(0,1)
    # u = torch.rand_like(logits)
    # # logiExp = torch.softmax(logits, -1)
    # # u = u
    # gumbels = -(-(u + EPSILON).log() - logits + logits.detach() + EPSILON).log()
    gumbels = (logits.detach() + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


# https://arxiv.org/pdf/1806.02867.pdf
# Direct Optimization through $\argmax$ for Discrete Variational Auto-Encoder
# Direct VAE
class DirectVAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
# def gumbel_perturbation(self,phi_x, eps=1e-10):
#     M,K,N = self.M,self.K,self.N

#     phi_x = phi_x.repeat(M,1)
#     shape = phi_x.size()
#     gumbel_noise = to_var(self.sample_gumbel(shape, eps=eps))
#     phi_x_gamma = phi_x + gumbel_noise
#     # hard:
#     _, k = phi_x_gamma.data.max(-1)

#     z_phi_gamma = to_var(torch.FloatTensor(*shape)).zero_().scatter_(-1, k.view(-1, 1), 1.0)

#     return z_phi_gamma,phi_x_gamma

# Relax / Rebar (Not working since grad-grad is hard to implement in ddp)
class DiscreteReparam(nn.Module):
    def __init__(self, coupled: bool = False):
        super().__init__()
        self._coupled = coupled

    @staticmethod
    def coupling(param: torch.Tensor, b: torch.Tensor, u: torch.Tensor):
        def robustGumbelcdf(g, K):
            return (EPSILON - (-g).exp() * (-K).exp()).exp() - EPSILON
        z = param - u.exponential_().log()
        vTop = robustGumbelcdf(z, -(param.logsumexp(dim=-1, keepdim=True)))
        topGumbel = (b * z).sum(dim=-1, keepdim=True)
        vRest = torch.exp(-torch.exp(param) * (torch.exp(-z) - torch.exp(-topGumbel)))
        return (1.0 - b) * vRest + b * vTop

    @staticmethod
    def gate(z: torch.Tensor):
        return F.one_hot(z.argmax(-1), z.shape[-1]).float()

    @staticmethod
    def categoricalForward(param: torch.Tensor, s: torch.Tensor, noise: torch.Tensor = None):
        if noise is not None:
            u = noise
        else:
            u = torch.rand_like(param)
        gumbel = -torch.empty_like(param, memory_format=torch.legacy_contiguous_format).exponential_().log()
        return param + gumbel

    @staticmethod
    def categoricalBackward(param:torch.Tensor, s: torch.Tensor, noise: torch.Tensor = None):
        def truncatedGumbel(gumbel, truncation):
            return -(EPSILON + (-gumbel).exp() + (-truncation).exp()).log()
        if noise is not None:
            v = noise
        else:
            v = torch.rand_like(param)
        gumbel = -v.exponential_().log()
        topGumbels = gumbel + param.logsumexp(dim=-1, keepdim=True)
        topGumbel = (s * topGumbels).sum(-1, keepdim=True)
        truncGumbel = truncatedGumbel(gumbel + param, topGumbel)
        return (1.0 - s) * truncGumbel + s * topGumbels

    @staticmethod
    def logpdf(param: torch.Tensor, b: torch.Tensor):
        return (param - param.logsumexp(dim=-1, keepdim=True) * b).sum()

    @staticmethod
    def softGate(z: torch.Tensor, t: torch.Tensor):
        return (z / t).softmax(-1)

    def forward(self, param, temperature):
        z = self.categoricalForward(param, torch.rand_like(param))
        b = self.gate(z).detach().requires_grad_()
        gatedZ = self.softGate(z, temperature)
        if self._coupled:
            v = self.coupling(param, b, torch.rand_like(param))
        else:
            v = torch.rand_like(param)
        v = v.detach().requires_grad_().clone()
        zb = self.categoricalBackward(param, b, v)
        gatedZb = self.softGate(zb, temperature)
        logP = self.logpdf(param, b)

        return b, gatedZ, gatedZb, logP

# Relax / Rebar (Not working since grad-grad is hard to implement in ddp)
class Relax(nn.Module):
    def __init__(self):
        super().__init__()
        self._ssim = MsSSIM(data_range=2, sizeAverage=True)

    def forward(self, real, restored, restoredZ, restoredZb, logit, logP, nu, regCoeff):
        loss = 1 - self._ssim(restored + 1, real + 1)
        lossZ = 1 - nu * self._ssim(restoredZ + 1, real + 1)
        lossZb = 1 - nu * self._ssim(restoredZb + 1, real + 1)

        approxGap = loss - 1. * lossZb

        # grad = torch.autograd.grad((logP * approxGap.detach()).mean() + loss.mean() + (lossZ - lossZb).mean(), model.parameters(), retain_graph=True, allow_unused=True)

        # allGrads = list()

        # for param, g1 in zip(model.parameters(), grad):
        #     g = torch.zeros_like(param)
        #     flags = False
        #     if g1 is not None:
        #         flags = True
        #         g += g1
        #     if flags:
        #         param.grad = g
                # allGrads.append(torch.flatten(g))

        # (logP * approxGap.detach()).mean().backward(create_graph=True, retain_graph=True)
        # loss.mean().backward(create_graph=True, retain_graph=True)
        # (lossZ - lossZb).mean().backward(create_graph=True, retain_graph=True)

        # allGrads = list()

        # allGrads = torch.cat(allGrads)
        # torch.autograd.backward(allGrads.mean(), [self._nu, temperature], retain_graph=True)
        # allGrads.mean().backward(retain_graph=True, inputs=[nu, temperature])

        posterior = Categorical(logits=logit)
        prior = Categorical(logits=torch.zeros_like(logit))
        reg = torch.distributions.kl_divergence(posterior, prior).mean()

        return ((logP * approxGap.detach()).mean() + loss.mean() + (lossZ - lossZb).mean() + (regCoeff * reg).mean()), F.mse_loss(restored, real) + F.l1_loss(restored, real), reg
