"""
model.py

Controlled via arch.yaml:
  norm:        batchnorm | groupnorm | instancenorm
  residual:    false | true
  anisotropic: false | true   → 3x3x1 kernels instead of 3x3x3
  channels:    [in, e1, e2, e3, e4, bottleneck]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Norm factory
# ---------------------------------------------------------------------------

def _make_norm(norm: str, n_channels: int) -> nn.Module:
    if norm == "batchnorm":
        return nn.BatchNorm3d(n_channels)
    elif norm == "groupnorm":
        groups = min(8, n_channels)
        while n_channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, n_channels)
    elif norm == "instancenorm":
        return nn.InstanceNorm3d(n_channels, affine=True)
    else:
        raise ValueError(f"Unknown norm: {norm}. Choose: batchnorm | groupnorm | instancenorm")


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def _conv_block(c_in: int, c_out: int, norm: str, dropout_prob: float,
                anisotropic: bool) -> nn.Sequential:
    """
    Double-conv block.
    anisotropic=True  → kernel (1,3,3), padding (0,1,1)  — no Z context
    anisotropic=False → kernel (3,3,3), padding (1,1,1)  — standard
    """
    k = (1, 3, 3) if anisotropic else (3, 3, 3)
    p = (0, 1, 1) if anisotropic else (1, 1, 1)

    layers = [
        nn.Conv3d(c_in,  c_out, k, padding=p),
        _make_norm(norm, c_out),
        nn.ReLU(inplace=True),
    ]
    if dropout_prob > 0:
        layers.append(nn.Dropout3d(dropout_prob))
    layers += [
        nn.Conv3d(c_out, c_out, k, padding=p),
        _make_norm(norm, c_out),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)


class _ResBlock(nn.Module):
    """Residual double-conv block with 1×1 projection if needed."""
    def __init__(self, c_in: int, c_out: int, norm: str,
                 dropout_prob: float, anisotropic: bool):
        super().__init__()
        self.conv = _conv_block(c_in, c_out, norm, dropout_prob, anisotropic)
        self.proj = nn.Conv3d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x) + self.proj(x), inplace=True)


def _make_block(c_in: int, c_out: int, norm: str, dropout_prob: float,
                residual: bool, anisotropic: bool) -> nn.Module:
    if residual:
        return _ResBlock(c_in, c_out, norm, dropout_prob, anisotropic)
    return _conv_block(c_in, c_out, norm, dropout_prob, anisotropic)


# ---------------------------------------------------------------------------
# UNet3D
# ---------------------------------------------------------------------------

class UNet3D(nn.Module):
    def __init__(self, channels: list, norm: str = "batchnorm",
                 dropout_prob: float = 0.3, residual: bool = False,
                 anisotropic: bool = False):
        """
        Args:
            channels:    [in, e1, e2, e3, e4, bottleneck]
            norm:        batchnorm | groupnorm | instancenorm
            residual:    use residual blocks
            anisotropic: use (1,3,3) kernels — recommended when Z spacing >> XY spacing
                         Your data: XY=0.1375µm/px, Z=0.5µm/plane → 3.6× anisotropy
        """
        super().__init__()
        ci, e1, e2, e3, e4, bn = channels
        b = lambda i, o: _make_block(i, o, norm, dropout_prob, residual, anisotropic)

        self.encoder1   = b(ci, e1)
        self.encoder2   = b(e1, e2)
        self.encoder3   = b(e2, e3)
        self.encoder4   = b(e3, e4)
        self.bottleneck = b(e4, bn)

        self.upconv4  = nn.ConvTranspose3d(bn, e4, 2, stride=2)
        self.decoder4 = b(bn, e4)
        self.upconv3  = nn.ConvTranspose3d(e4, e3, 2, stride=2)
        self.decoder3 = b(e4, e3)
        self.upconv2  = nn.ConvTranspose3d(e3, e2, 2, stride=2)
        self.decoder2 = b(e3, e2)
        self.upconv1  = nn.ConvTranspose3d(e2, e1, 2, stride=2)
        self.decoder1 = b(e2, e1)

        self.final_conv = nn.Conv3d(e1, 1, 1)

    def _dynamic_pool(self, x: torch.Tensor) -> torch.Tensor:
        kernel_z = 2 if x.shape[2] > 1 else 1
        return F.max_pool3d(x, kernel_size=(kernel_z, 2, 2), stride=(kernel_z, 2, 2))

    @staticmethod
    def _upsample_cat(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if up.shape != skip.shape:
            up = F.interpolate(up, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return torch.cat([up, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(self._dynamic_pool(e1))
        e3 = self.encoder3(self._dynamic_pool(e2))
        e4 = self.encoder4(self._dynamic_pool(e3))
        b  = self.bottleneck(self._dynamic_pool(e4))

        d4 = self.decoder4(self._upsample_cat(self.upconv4(b),  e4))
        d3 = self.decoder3(self._upsample_cat(self.upconv3(d4), e3))
        d2 = self.decoder2(self._upsample_cat(self.upconv2(d3), e2))
        d1 = self.decoder1(self._upsample_cat(self.upconv1(d2), e1))

        return self.final_conv(d1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg) -> nn.Module:
    return UNet3D(
        channels=cfg.channels,
        norm=cfg.norm,
        dropout_prob=cfg.dropout_prob,
        residual=cfg.residual,
        anisotropic=getattr(cfg, "anisotropic", False),
    )
