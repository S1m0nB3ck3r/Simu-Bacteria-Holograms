# -*- coding: utf-8 -*-

"""
Filename: losses.py

Description:
Fonctions de coût testées pour le U-Net 3D. Toutes prennent des logits (B, 1, Z, Y, X) et une cible
de même forme. Deux familles, selon la cible :
  Masque du bâtonnet (cible {0, 1}, ~10⁻⁴ de voxels positifs) :
    - 'bce_dice'      : BCE (poids des positifs) + Dice doux — la loss de l'ancien code ;
    - 'focal_tversky' : focale (γ = 2) + Tversky focale (α = 0,3 sur les faux positifs, β = 0,7 sur
                        les faux négatifs : favorise le rappel des objets faibles) ;
  Carte de chaleur des centres (gaussienne, 1 au centre) :
    - 'centernet'     : focale à pénalité réduite de CenterNet (Zhou et al. 2019), normalisée par le
                        nombre de centres ;
    - 'wmse'          : erreur quadratique pondérée (poids 1 + w · cible) sur la sigmoïde.

Author: Simon BECKER
Date: 2026-10-17

License:
GNU General Public License v3.0
"""

import torch
import torch.nn.functional as F

TARGET_OF_LOSS = {'bce_dice': 'mask', 'focal_tversky': 'mask', 'centernet': 'heat', 'wmse': 'heat'}


def bce_dice(logits, target, pos_weight=20.0, dice_weight=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(pos_weight, device=logits.device))
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    dice = 1 - (2 * inter + 1.0) / (p.sum() + target.sum() + 1.0)
    return bce + dice_weight * dice


def focal_tversky(logits, target, gamma=2.0, alpha_f=0.75, alpha=0.3, beta=0.7, gamma_t=0.75):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt = p * target + (1 - p) * (1 - target)
    w = alpha_f * target + (1 - alpha_f) * (1 - target)
    focal = (w * (1 - pt) ** gamma * ce).sum() / torch.clamp(target.sum(), min=1.0)
    tp = (p * target).sum()
    fp = (p * (1 - target)).sum()
    fn = ((1 - p) * target).sum()
    tversky = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    return focal + (1 - tversky) ** gamma_t


def centernet(logits, target, alpha=2.0, beta=4.0):
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.ge(0.999).float()
    log_p = F.logsigmoid(logits)
    log_1mp = F.logsigmoid(-logits)
    pos_loss = -((1 - p) ** alpha * log_p * pos).sum()
    neg_loss = -((1 - target) ** beta * p ** alpha * log_1mp * (1 - pos)).sum()
    return (pos_loss + neg_loss) / torch.clamp(pos.sum(), min=1.0)


def wmse(logits, target, w=100.0):
    p = torch.sigmoid(logits)
    weight = 1.0 + w * target
    return (weight * (p - target) ** 2).sum() / weight.sum()


LOSSES = {'bce_dice': bce_dice, 'focal_tversky': focal_tversky, 'centernet': centernet, 'wmse': wmse}
