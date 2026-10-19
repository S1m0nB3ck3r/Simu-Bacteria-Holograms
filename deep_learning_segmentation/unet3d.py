# -*- coding: utf-8 -*-

"""
Filename: unet3d.py

Description:
U-Net 3D résiduel pour la détection / segmentation de bactéries dans le volume reconstruit.
  - entrée : 2 canaux (Re, Im du champ reconstruit, en unités de bruit capteur), (B, 2, Z, Y, X) ;
  - blocs résiduels conv 3³ – normalisation – LeakyReLU. BatchNorm par défaut : ses statistiques
    portent sur les millions de voxels de chaque patch (pas sur les 4 patchs du lot), et à
    l'inférence elle se replie dans la convolution qui la précède (fold_batchnorm) : coût nul.
    GroupNorm en option ('group') : mesuré 2,4 fois plus lent en inférence (619 contre 255 ms par
    tuile 104 × 320 × 320, Quadro RTX 3000, fp16) ;
  - `depth` sous-échantillonnages par 2 (max-pool), remontée par convolution transposée, liaisons
    par concaténation ; tailles d'entrée multiples de 2^depth ;
  - sortie : 1 canal de logits (masque du bâtonnet ou carte de chaleur des centres, selon la loss).
Voxel de 0,35 × 0,35 × 1 µm : la réponse d'une bactérie (largeurs à mi-hauteur 2,1 µm en XY, 5,3 µm
en Z) fait ~6 voxels dans chaque direction — noyaux isotropes.

Author: Simon BECKER
Date: 2026-10-17

License:
GNU General Public License v3.0
"""

import torch
import torch.nn as nn


def make_norm(kind, ch):
    if kind == 'group':
        return nn.GroupNorm(min(8, ch // 4) if ch >= 8 else 1, ch)
    return nn.BatchNorm3d(ch)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, dropout=0.0, norm='batch'):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1, bias=False)
        self.n1 = make_norm(norm, cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.n2 = make_norm(norm, cout)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Identity() if cin == cout else nn.Conv3d(cin, cout, 1, bias=False)

    def forward(self, x):
        y = self.act(self.n1(self.conv1(x)))
        y = self.drop(y)
        y = self.n2(self.conv2(y))
        return self.act(y + self.skip(x))


class UNet3D(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base=16, depth=3, dropout=0.0, norm='batch'):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.depth = depth
        self.enc = nn.ModuleList([ResBlock(in_ch, ch[0], norm=norm)] +
                                 [ResBlock(ch[i - 1], ch[i], dropout if i == depth else 0.0, norm=norm)
                                  for i in range(1, depth + 1)])
        self.pool = nn.MaxPool3d(2)
        self.up = nn.ModuleList([nn.ConvTranspose3d(ch[i], ch[i - 1], 2, stride=2) for i in range(depth, 0, -1)])
        self.dec = nn.ModuleList([ResBlock(2 * ch[i - 1], ch[i - 1], norm=norm) for i in range(depth, 0, -1)])
        self.head = nn.Conv3d(ch[0], out_ch, 1)
        # biais initial : probabilité a priori faible (objets rares), évite l'effondrement initial
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else self.pool(x))
            skips.append(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips[:-1])):
            x = dec(torch.cat([up(x), s], dim=1))
        return self.head(x)


@torch.no_grad()
def fold_batchnorm(model):
    """Inférence : replie chaque BatchNorm dans la convolution qui la précède (modèle en eval)."""
    for blk in model.modules():
        if isinstance(blk, ResBlock):
            for cname, nname in (('conv1', 'n1'), ('conv2', 'n2')):
                conv, bn = getattr(blk, cname), getattr(blk, nname)
                if not isinstance(bn, nn.BatchNorm3d):
                    continue
                scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                fused = nn.Conv3d(conv.in_channels, conv.out_channels, 3, padding=1, bias=True).to(conv.weight.device)
                fused.weight.copy_(conv.weight * scale[:, None, None, None, None])
                fused.bias.copy_(bn.bias - bn.running_mean * scale)
                setattr(blk, cname, fused)
                setattr(blk, nname, nn.Identity())
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
