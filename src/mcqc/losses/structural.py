import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from pytorch_msssim import ms_ssim

from mcqc import Consts


class QError(nn.Module):
    def forward(self, latents, codebooks, logits, codes):
        loss = 0.0
        for z, c, l, b in zip(latents, codebooks, logits, codes):
            z = z.detach().permute(0, 2, 3, 1)
            k = l.shape[-1]
            soft = l @ c
            softQE = F.mse_loss(soft, z)
            oneHot = F.one_hot(b, k).float()
            hard = oneHot @ c
            hardQE = F.mse_loss(hard, z)
            loss += (softQE + hardQE + 0.1 * F.mse_loss(hard, soft)).mean()
        return loss

class CompressionLoss(nn.Module):
    def forward(self, images, restored, codes, latents, logits, quantizeds, cv):
        l2Loss = F.mse_loss(restored, images)
        l1Loss = F.l1_loss(restored, images)
        ssimLoss = 1 - ms_ssim((restored + 1), (images + 1), data_range=2.0)

        regs = list()
        for logit, q, latent in zip(logits, quantizeds, latents):
            # N, H, W, K -> NHW, K
            unNormlogit = logit.reshape(len(logit), -1, logit.shape[-1])
            reg = compute_penalties(unNormlogit, individual_entropy_coeff=0.01, allowed_js=4.0, js_coeff=0.01, cv_coeff=cv, eps=Consts.Eps)
            regs.append(reg)
        regs = sum(regs)
        return ssimLoss, l1Loss + l2Loss, regs


def p2pJSDivLoss(probP, probQ, allowed_js, eps=1e-9):
    mean = (probP + probQ) / 2
    jsEstimation = (F.kl_div(probP.log(), mean, reduction="none") + F.kl_div(probQ.log(), mean, reduction="none")) / 2
    jsEstimation = F.relu(allowed_js - jsEstimation + eps)
    jsEstimation = jsEstimation[jsEstimation > eps]
    return jsEstimation.mean() if len(jsEstimation > 0) else 0.0


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
    distributions = Categorical(logits=logits.reshape(-1, logits.shape[-1]))
    minSingle = individual_entropy_coeff * distributions.entropy().mean()

    p = torch.softmax(logits, dim=-1)

    shuffleIdx = torch.randperm(logits.shape[1])
    half = logits.shape[1] // 2
    jsEstimation = p2pJSDivLoss(p[:, shuffleIdx[:half]], p[:, shuffleIdx[half:]], allowed_js, eps)

    # p = p.reshape(-1, logits.shape[-1])
    load = torch.mean(p, dim=0)  # [..., codebook_size]
    mean = load.mean()
    variance = torch.mean((load - mean) ** 2)
    cvPenalty = variance / (mean ** 2 + eps)
    return minSingle + js_coeff * jsEstimation + cv_coeff * cvPenalty
