import torch
from torch import nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

from mcqc import Consts


class CompressionLoss(nn.Module):
    def forward(self, images, restored, codes, latents, logits, quantizeds):
        l2Loss = F.mse_loss(restored, images)
        l1Loss = F.l1_loss(restored, images)
        ssimLoss = 1 - ms_ssim((restored + 1), (images + 1), data_range=2.0)

        regs = list()
        for logit, q, latent in zip(logits, quantizeds, latents):
            # N, H, W, K -> NHW, K
            unNormlogit = logit.reshape(-1, logit.shape[-1])
            maxGlobal = compute_penalties(unNormlogit, cv_coeff=0.02, eps=Consts.Eps)
            regs.append(maxGlobal)
        regs = sum(regs)
        return ssimLoss, l1Loss + l2Loss, regs


def compute_penalties(logits, individual_entropy_coeff=0.0, allowed_entropy=0.0, global_entropy_coeff=0.0,
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
    p = torch.softmax(logits, dim=-1)
    # logp = torch.log_softmax(logits, dim=-1)
    # [batch_size, ..., codebook_size]

    # individual_entropy_values = -torch.sum(p * logp, dim=-1)
    # clipped_entropy = torch.nn.functional.relu(allowed_entropy - individual_entropy_values + eps).mean()
    # individual_entropy = (individual_entropy_values.mean() - clipped_entropy).detach() + clipped_entropy

    # global_p = torch.mean(p, dim=0)  # [..., codebook_size]
    # global_logp = torch.logsumexp(logp, dim=0) - np.log(float(logp.shape[0]))  # [..., codebook_size]
    # global_entropy = -torch.sum(global_p * global_logp, dim=-1).mean()

    load = torch.mean(p, dim=0)  # [..., codebook_size]
    mean = load.mean()
    variance = torch.mean((load - mean) ** 2)
    cvPenalty = variance / (mean ** 2 + eps)
    return cv_coeff * cvPenalty
