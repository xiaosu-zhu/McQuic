import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, OneHotCategorical

from mcqc import Consts
from .ssim import ms_ssim


class QError(nn.Module):
    def forward(self, latents, zqs, softs):
        loss = 0.0
        for z, zq, soft in zip(latents, zqs, softs):
            qe = F.mse_loss(z.detach(), zq, reduction='none').mean(axis=(0, 2))
            commit = F.mse_loss(z, zq.detach(), reduction='none').mean(axis=(0, 2))
            softQE = F.mse_loss(z.detach(), soft, reduction='none').mean(axis=(0, 2))
            softCommit = F.mse_loss(z, soft.detach(), reduction='none').mean(axis=(0, 2))
            joint = F.mse_loss(soft, zq, reduction='none').mean(axis=(0, 2))
            loss += qe + softQE + 0.1 * joint # + 0.01 * commit + 0.1 * (softQE + 0.01 * softCommit)
        return loss


class CompressionLoss(nn.Module):
    def forward(self, images, restored, latents, logits, quantizeds, cv):
        l2Loss = F.mse_loss(restored, images, reduction='none').mean(axis=(1, 2, 3))
        l1Loss = F.l1_loss(restored, images, reduction='none').mean(axis=(1, 2, 3))
        ssimLoss = 1 - ms_ssim((restored + 1), (images + 1), data_range=2.0, size_average=False)

        regs = list()
        if logits is not None:
            for logit in logits:
                # N, H, W, K -> N, HW, K
                unNormlogit = logit.reshape(len(logit), -1, logit.shape[-1])

                # [n, k]
                summedLogit = unNormlogit.mean(1)

                posterior = OneHotCategorical(logits=summedLogit, validate_args=False)
                prior = OneHotCategorical(probs=torch.ones_like(summedLogit) / summedLogit.shape[-1], validate_args=False)
                reg = cv * torch.distributions.kl_divergence(posterior, prior)
                # reg = compute_penalties(unNormlogit, allowed_entropy=0.1, individual_entropy_coeff=cv, allowed_js=4.0, js_coeff=cv, cv_coeff=cv, eps=Consts.Eps)
                regs.append(reg)
            regs = sum(regs)
        return ssimLoss, l1Loss + l2Loss, regs # + 10 * stdReg


class CompressionLossTwoStage(nn.Module):
    def forward(self, images, restored, latents, logits, quantizeds, cv, e2e):
        l2Loss = F.mse_loss(restored, images, reduction='none').mean(axis=(1, 2, 3))
        l1Loss = F.l1_loss(restored, images, reduction='none').mean(axis=(1, 2, 3))
        ssimLoss = 1 - ms_ssim((restored + 1), (images + 1), data_range=2.0, size_average=False)

        l2QLoss = list()
        l1QLoss = list()
        if not e2e:
            for latent, q in zip(latents, quantizeds):
                l2QLoss.append(F.mse_loss(latent.detach(), q, reduction='none').mean(axis=(1, 2, 3)))
                l1QLoss.append(F.l1_loss(latent.detach(), q, reduction='none').mean(axis=(1, 2, 3)))
                # l2QLoss.append(0.00001 * F.mse_loss(latent, q.detach(), reduction='none').mean(axis=(1, 2, 3)))
                # l1QLoss.append(0.00001 * F.l1_loss(latent, q.detach(), reduction='none').mean(axis=(1, 2, 3)))

        l1QLoss = sum(l1QLoss)
        l2QLoss = sum(l2QLoss)

        regs = list()
        if logits is not None:
            for logit in logits:
                # N, H, W, K -> N, HW, K
                batchWiseLogit = logit.reshape(len(logit), -1, logit.shape[-1])

                # [n, k]
                # summedProb = batchWiseLogit.mean(1).sigmoid()

                # target = torch.ones_like(summedProb) / 2.0
                # [n, ]
                # reg = F.binary_cross_entropy(summedProb, target, reduction='none').sum(-1)

                # var = batchWiseLogit.var(1).sum(-1)

                # [n, k] -> [n, ]
                # diversity = torch.minimum(var, torch.ones_like(var))
                # reg -= diversity

                diversity = batchWiseLogit.std(1).mean(-1).sigmoid()

                # summedProb = batchWiseLogit.sum(1)
                # posterior = OneHotCategorical(logits=summedProb)
                # prior = OneHotCategorical(probs=torch.ones_like(summedProb) / summedProb.shape[-1])
                # reg = torch.distributions.kl_divergence(posterior, prior) / diversity
                reg = compute_penalties(batchWiseLogit, allowed_entropy=0.1, individual_entropy_coeff=cv, allowed_js=4.0, js_coeff=cv, cv_coeff=cv, eps=Consts.Eps)
                reg = reg / diversity
                regs.append(reg)
            regs = sum(regs)
        return ssimLoss, l1Loss + l2Loss, l1QLoss + l2QLoss, regs # + 10 * stdReg


class CompressionReward(nn.Module):
    def forward(self, images, restored):
        l2Loss = F.mse_loss(restored, images, reduction='none').mean(axis=(1, 2, 3))
        l1Loss = F.l1_loss(restored, images, reduction='none').mean(axis=(1, 2, 3))
        ssimLoss = 1 - ms_ssim((restored + 1), (images + 1), data_range=2.0, size_average=False)

        ssim = 20 * (1.0 / ssimLoss.detach().sqrt()).log10()
        psnr = 20 * (2.0 / l2Loss.detach().sqrt()).log10()

        return ssim, psnr, ssimLoss, l1Loss + l2Loss


def p2pJSDivLoss(probP, probQ, allowed_js, eps=1e-9):
    mean = (probP + probQ) / 2
    jsEstimation = (F.kl_div(probP.log(), mean, reduction="none") + F.kl_div(probQ.log(), mean, reduction="none")) / 2
    jsEstimation = F.relu(allowed_js - jsEstimation + eps)
    jsEstimation[jsEstimation > eps] *= 0.0
    # rank = len(jsEstimation.shape)
    return jsEstimation.mean(axis=(1, 2))


def compute_penalties(logits, allowed_entropy=0.1, individual_entropy_coeff=0.0, allowed_js=0.0, global_entropy_coeff=0.0, js_coeff=0.0,
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
    # # [batch_size, ..., codebook_size]

    # # [N, h*w]
    # individual_entropy_values = -torch.sum(p * logp, dim=-1)
    # individual_entropy = individual_entropy_values.mean(-1)
    # clipped_entropy = torch.nn.functional.relu(allowed_entropy - individual_entropy_values + eps).mean()
    # individual_entropy = (individual_entropy_values.mean() - clipped_entropy).detach() + clipped_entropy

    # global_p = torch.mean(p, dim=0)  # [..., codebook_size]
    # global_logp = torch.logsumexp(logp, dim=0) - np.log(float(logp.shape[0]))  # [..., codebook_size]
    # global_entropy = -torch.sum(global_p * global_logp, dim=-1).mean()
    '''
    distributions = Categorical(logits=logits.reshape(-1, logits.shape[-1]))
    minSingle = individual_entropy_coeff * distributions.entropy().mean()
    '''

    # shuffleIdx = torch.randperm(logits.shape[1])
    # half = logits.shape[1] // 2
    # jsEstimation = p2pJSDivLoss(p[:, shuffleIdx[:half]], p[:, shuffleIdx[half:]], allowed_js, eps)

    p = torch.softmax(logits, dim=-1)
    # p = p.reshape(-1, logits.shape[-1])
    # [N, K]
    load = torch.mean(p, dim=1)  # [N, codebook_size]
    # [N, ]
    mean = load.mean(-1)
    # [N, ]
    variance = ((load - mean[:, None]) ** 2).mean(-1)
    cvPenalty = variance / (mean ** 2 + eps)
    return cv_coeff * cvPenalty
