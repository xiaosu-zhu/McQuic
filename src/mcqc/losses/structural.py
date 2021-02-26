import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from pytorch_msssim import ms_ssim

from mcqc import Consts


class QError(nn.Module):
    def forward(self, latents, zqs, softs):
        loss = 0.0
        for z, zq, soft in zip(latents, zqs, softs):
            qe = F.mse_loss(z.detach(), zq, reduction='none').mean(axis=(0, 2))
            commit = F.mse_loss(z, zq.detach(), reduction='none').mean(axis=(0, 2))
            softQE = F.mse_loss(z.detach(), soft, reduction='none').mean(axis=(0, 2))
            softCommit = F.mse_loss(z, soft.detach(), reduction='none').mean(axis=(0, 2))
            # joint = F.mse_loss(soft, zq, reduction='none').mean(axis=(0, 2))
            loss += qe + 0.01 * commit + 0.1 * (softQE + 0.01 * softCommit)
        return loss

class CompressionLoss(nn.Module):
    def forward(self, images, restored, codebooks, latents, logits, quantizeds, cv):
        l2Loss = F.mse_loss(restored, images, reduction='none').mean(axis=(1, 2, 3))
        l1Loss = F.l1_loss(restored, images, reduction='none').mean(axis=(1, 2, 3))
        ssimLoss = 1 - ms_ssim((restored + 1), (images + 1), data_range=2.0, size_average=False)

        regs = list()
        if logits is not None:
            for logit, q, latent in zip(logits, quantizeds, latents):
                # N, H, W, K -> NHW, K
                unNormlogit = logit.reshape(len(logit), -1, logit.shape[-1])
                reg = compute_penalties(unNormlogit, individual_entropy_coeff=0.0, allowed_js=4.0, js_coeff=0.001, cv_coeff=cv, eps=Consts.Eps)
                regs.append(reg)
            regs = sum(regs)
            stdReg = 0.0
            for codebook in codebooks:
                stdReg += ((codebook.std(0) - 1) ** 2).mean()

        return ssimLoss, l1Loss + l2Loss, regs # + 10 * stdReg


def p2pJSDivLoss(probP, probQ, allowed_js, eps=1e-9):
    mean = (probP + probQ) / 2
    jsEstimation = (F.kl_div(probP.log(), mean, reduction="none") + F.kl_div(probQ.log(), mean, reduction="none")) / 2
    jsEstimation = F.relu(allowed_js - jsEstimation + eps)
    jsEstimation[jsEstimation > eps] *= 0.0
    # rank = len(jsEstimation.shape)
    return jsEstimation.mean(axis=(1, 2))


def compute_penalties(logits, individual_entropy_coeff=0.0, allowed_js=0.0, global_entropy_coeff=0.0, js_coeff=0.0,
                      cv_coeff=0.0, eps=1e-9):
    """
    Computes typical regularizers for gumbel-softmax quantization
    Regularization is of slight help when performing hard quantization, but it isn't critical
    :param logits: tensor [batch_size, ..., codebook_size]
    :param individual_entropy_coeff: penalizes mean individual entropy
    :param allowed_entropy: does not penalize individual_entropy if it is below this value
    :param cv_coeff: penalizes squared coefficient of variation
    :param global_entropy_coeff: coefficient for entropy of mean probabilities over batch
        this value should typically be negative (e.g. -1), works similar to cv_coeff
    """
    # logp = torch.log_softmax(logits, dim=-1)
    # [batch_size, ..., codebook_size]

    # individual_entropy_values = -torch.sum(p * logp, dim=-1)
    # clipped_entropy = torch.nn.functional.relu(allowed_entropy - individual_entropy_values + eps).mean()
    # individual_entropy = (individual_entropy_values.mean() - clipped_entropy).detach() + clipped_entropy

    # global_p = torch.mean(p, dim=0)  # [..., codebook_size]
    # global_logp = torch.logsumexp(logp, dim=0) - np.log(float(logp.shape[0]))  # [..., codebook_size]
    # global_entropy = -torch.sum(global_p * global_logp, dim=-1).mean()
    '''
    distributions = Categorical(logits=logits.reshape(-1, logits.shape[-1]))
    minSingle = individual_entropy_coeff * distributions.entropy().mean()
    '''
    p = torch.softmax(logits, dim=-1)

    shuffleIdx = torch.randperm(logits.shape[1])
    half = logits.shape[1] // 2
    jsEstimation = p2pJSDivLoss(p[:, shuffleIdx[:half]], p[:, shuffleIdx[half:]], allowed_js, eps)

    # p = p.reshape(-1, logits.shape[-1])
    load = torch.mean(p, dim=1)  # [N, codebook_size]
    mean = load.mean(-1)
    variance = ((load - mean[:, None]) ** 2).mean(-1)
    cvPenalty = variance / (mean ** 2 + eps)
    return js_coeff * jsEstimation + cv_coeff * cvPenalty
