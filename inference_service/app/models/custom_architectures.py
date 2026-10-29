"""Original (training-time) architectures that are NOT standard segmentation-models-pytorch classes.

WHY THIS FILE EXISTS
--------------------
The MA/HE checkpoints use a custom DeepLabV3+ architecture whose decoder is structured as:
  aspp.branches.N  (4 parallel conv+BN branches)
  aspp.pool        (global average pool branch)
  aspp.project     (1×1 projection over concatenated branches)
  low              (1×1 low-level feature projection)
  decoder.0 / .1  (two sequential conv+BN blocks)
  head             (1×1 classification conv)

The EX checkpoint uses a custom U-Net architecture whose decoder is structured as:
  up4 / up3 / up2 / up1 / up0  (each: two Conv-BN-ReLU double-conv blocks)
  head                          (1×1 classification conv)

The SE checkpoint uses a custom U-Net++ architecture whose decoder is structured as:
  blocks.x_i_j   (each: two Conv-BN-ReLU double-conv blocks, with aggregated skip inputs)
  final_block    (last double-conv block before classification)
  head           (1×1 classification conv)

All four checkpoints also contain encoder.* keys from a standard SMP ResNet encoder.
Standard smp.DeepLabV3Plus / smp.Unet / smp.UnetPlusPlus use different decoder module names,
so loading with strict=True fails -- and loading with strict=False would silently produce garbage.

The classes below were reconstructed from checkpoint key/shape data produced by
scripts/inspect_checkpoints.py.  Each class includes ``self.encoder`` (a real SMP encoder
so that the ``encoder.*`` checkpoint keys are accepted) plus the custom decoder.

HOW TO ADD OR MODIFY
---------------------
1. Run:  py scripts\\inspect_checkpoints.py   (prints key prefixes and shapes per checkpoint)
2. Paste or edit the training class(es) here (same attribute names -> same state_dict keys).
3. Register a builder that returns an *uninitialised* model (no pretrained weights download):

    CUSTOM_BUILDERS["CODE"] = lambda meta: MyClass()

The service loads the checkpoint with strict=True. If keys still do not match, the lesion is
reported as "unavailable" with a key-mismatch summary instead of returning fake masks.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp

# lesion code ("MA" | "HE" | "EX" | "SE") -> callable(meta: dict) -> nn.Module
CUSTOM_BUILDERS: dict[str, Callable[[dict], nn.Module]] = {}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_encoder(encoder_name: str) -> nn.Module:
    """Return an uninitialised SMP encoder whose state_dict keys match the checkpoint."""
    # Build a throwaway SMP model just to steal its encoder
    tmp = smp.Unet(encoder_name=encoder_name, encoder_weights=None,
                   in_channels=3, classes=1)
    return tmp.encoder


# ---------------------------------------------------------------------------
# MA / HE  --  Custom DeepLabV3+
#
# Checkpoint key structure (non-encoder part):
#   aspp.branches.0  – 1×1 conv + BN  (256, 2048, 1, 1)
#   aspp.branches.1  – 3×3 dilated conv + BN  (rate 6)
#   aspp.branches.2  – 3×3 dilated conv + BN  (rate 12)
#   aspp.branches.3  – 3×3 dilated conv + BN  (rate 18)
#   aspp.pool.1      – 1×1 conv after global avg pool  (256, 2048, 1, 1)
#   aspp.pool.2      – BN(256)
#   aspp.project.0   – 1×1 conv merging 5 branches  (256, 1280, 1, 1)
#   aspp.project.1   – BN(256)
#   low.0            – 1×1 conv on low-level features  (48, 256, 1, 1)
#   low.1            – BN(48)
#   decoder.0        – 3×3 conv  (256, 304, 3, 3)
#   decoder.1        – 3×3 conv  (256, 256, 3, 3)
#   head             – 1×1 conv  (1, 256, 1, 1)
# ---------------------------------------------------------------------------

class _ASPPBranch(nn.Module):
    """Single ASPP branch: Conv2d (optionally dilated) + BN + ReLU.

    Module indices must match checkpoint: 0 = Conv2d, 1 = BatchNorm2d.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        pad = dilation if kernel == 3 else 0
        self.add_module("0", nn.Conv2d(in_ch, out_ch, kernel,
                                       padding=pad, dilation=dilation, bias=False))
        self.add_module("1", nn.BatchNorm2d(out_ch))
        self._relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._relu(self._modules["1"](self._modules["0"](x)))


class _ASPPPool(nn.Module):
    """Global average pool branch of ASPP.

    Checkpoint keys under ``aspp.pool``:
      .1  -> Conv2d        (index 1, after AdaptiveAvgPool2d at index 0)
      .2  -> BatchNorm2d   (index 2)

    The indices must live directly on this module (not nested under self.pool)
    so that state_dict keys are ``aspp.pool.1.*`` not ``aspp.pool.pool.1.*``.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # index 0 = AdaptiveAvgPool2d (no params – not in checkpoint)
        self.add_module("0", nn.AdaptiveAvgPool2d(1))
        # index 1 = Conv2d
        self.add_module("1", nn.Conv2d(in_ch, out_ch, 1, bias=False))
        # index 2 = BatchNorm2d
        self.add_module("2", nn.BatchNorm2d(out_ch))
        self._relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        out = self._relu(self._modules["2"](self._modules["1"](self._modules["0"](x))))
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)


class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling matching the training checkpoint."""

    def __init__(self, in_ch: int = 2048, out_ch: int = 256,
                 rates: tuple = (6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            _ASPPBranch(in_ch, out_ch, kernel=1, dilation=1),         # 0
            _ASPPBranch(in_ch, out_ch, kernel=3, dilation=rates[0]),  # 1
            _ASPPBranch(in_ch, out_ch, kernel=3, dilation=rates[1]),  # 2
            _ASPPBranch(in_ch, out_ch, kernel=3, dilation=rates[2]),  # 3
        ])
        self.pool = _ASPPPool(in_ch, out_ch)
        n_concat = out_ch * (len(self.branches) + 1)  # 256 * 5 = 1280
        # project: index 0 = Conv2d, index 1 = BN
        self.project = nn.Sequential(
            nn.Conv2d(n_concat, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self._relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [b(x) for b in self.branches] + [self.pool(x)]
        return self._relu(self.project(torch.cat(parts, dim=1)))


class TrainingDeepLabV3Plus(nn.Module):
    """Custom DeepLabV3+ used during training for MA and HE checkpoints.

    Includes ``self.encoder`` (SMP ResNet-50) so that the ``encoder.*``
    checkpoint keys are accepted by strict load.
    """

    def __init__(self, encoder_name: str = "resnet50", encoder_in_ch: int = 2048,
                 low_ch: int = 256, low_out_ch: int = 48,
                 dec_ch: int = 256, num_classes: int = 1):
        super().__init__()
        self.encoder = _get_encoder(encoder_name)

        self.aspp = _ASPP(in_ch=encoder_in_ch, out_ch=dec_ch)

        # low-level feature projection: index 0 = Conv2d, index 1 = BN
        self.low = nn.Sequential(
            nn.Conv2d(low_ch, low_out_ch, 1, bias=False),
            nn.BatchNorm2d(low_out_ch),
        )

        concat_ch = dec_ch + low_out_ch  # 256 + 48 = 304
        # decoder: two conv blocks stored as decoder.0 and decoder.1
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(concat_ch, dec_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(dec_ch),
            ),
            nn.Sequential(
                nn.Conv2d(dec_ch, dec_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(dec_ch),
            ),
        ])
        self.head = nn.Conv2d(dec_ch, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0, w0 = x.shape[-2], x.shape[-1]
        feats = self.encoder(x)
        low = feats[2]
        high = feats[-1]

        aspp_out = self.aspp(high)
        aspp_up = F.interpolate(aspp_out, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low_proj = F.relu(self.low(low), inplace=True)

        d = torch.cat([aspp_up, low_proj], dim=1)
        d = F.relu(self.decoder[0](d), inplace=True)
        d = F.relu(self.decoder[1](d), inplace=True)

        logits = self.head(d)
        return F.interpolate(logits, size=(h0, w0), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# EX  --  Custom U-Net
#
# Checkpoint key structure (non-encoder part):
#   up4.conv.0.0 / .0.1   Conv+BN   (256, 768, 3, 3) / (256,)
#   up4.conv.1.0 / .1.1   Conv+BN   (256, 256, 3, 3) / (256,)
#   up3 … up0  (halving channels each step)
#   head   Conv2d  (1, 16, 1, 1)
#
# ResNet-34 encoder output channels per stage:
#   stem=64, layer1=64, layer2=128, layer3=256, layer4=512
# ---------------------------------------------------------------------------

class _DoubleConvBlock(nn.Module):
    """Two Conv-BN blocks whose keys appear as <parent>.conv.0.* and .1.*

    conv is an nn.ModuleList so checkpoint indices map: conv.0 and conv.1.
    Each entry is an nn.Sequential so sub-indices map: .0 = Conv2d, .1 = BN.
    """
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid_ch),
            ),
            nn.Sequential(
                nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
            ),
        ])
        self._relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._relu(self.conv[0](x))
        return self._relu(self.conv[1](x))


class _UpBlock(nn.Module):
    """Upsample + skip-cat + double-conv.

    Checkpoint keys: ``upN.conv.0.*`` and ``upN.conv.1.*``
    So ``self.conv`` must be an nn.ModuleList directly (not a wrapper module).
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        total_in = in_ch + skip_ch
        # conv is an nn.ModuleList so state_dict keys are conv.0.* and conv.1.*
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(total_in, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
            ),
            nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
            ),
        ])
        self._relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self._relu(self.conv[0](x))
        return self._relu(self.conv[1](x))


class TrainingUNet(nn.Module):
    """Custom U-Net used during training for the EX checkpoint.

    Includes ``self.encoder`` (SMP ResNet-34) so that ``encoder.*``
    checkpoint keys are accepted by strict load.

    Observed checkpoint shapes:
      up4.conv.0.0.weight: (256, 768, 3, 3)  in=512+256=768
      up3.conv.0.0.weight: (128, 384, 3, 3)  in=256+128=384
      up2.conv.0.0.weight: ( 64, 192, 3, 3)  in=128+ 64=192
      up1.conv.0.0.weight: ( 32, 128, 3, 3)  in= 64+ 64=128
      up0.conv.0.0.weight: ( 16,  32, 3, 3)  in= 32+  0= 32
      head.weight:          (  1,  16, 1, 1)
    """

    def __init__(self, encoder_name: str = "resnet34", num_classes: int = 1):
        super().__init__()
        self.encoder = _get_encoder(encoder_name)

        self.up4 = _UpBlock(in_ch=512, skip_ch=256, out_ch=256)
        self.up3 = _UpBlock(in_ch=256, skip_ch=128, out_ch=128)
        self.up2 = _UpBlock(in_ch=128, skip_ch=64,  out_ch=64)
        self.up1 = _UpBlock(in_ch=64,  skip_ch=64,  out_ch=32)
        self.up0 = _UpBlock(in_ch=32,  skip_ch=0,   out_ch=16)
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0, w0 = x.shape[-2], x.shape[-1]
        feats = self.encoder(x)
        d4 = self.up4(feats[5], feats[4])
        d3 = self.up3(d4, feats[3])
        d2 = self.up2(d3, feats[2])
        d1 = self.up1(d2, feats[1])

        d1_up = F.interpolate(d1, size=(h0, w0), mode="bilinear", align_corners=False)
        d0 = self.up0(d1_up)
        return self.head(d0)


# ---------------------------------------------------------------------------
# SE  --  Custom U-Net++
#
# Checkpoint key structure (non-encoder part):
#   blocks.x_i_j.0.{0=Conv, 1=BN}   first  conv in double-conv
#   blocks.x_i_j.1.{0=Conv, 1=BN}   second conv in double-conv
#   final_block.0.{0=Conv, 1=BN}
#   final_block.1.{0=Conv, 1=BN}
#   head  Conv2d (1, 16, 1, 1)
#
# ResNet-34 encoder channels: [64, 64, 128, 256, 512]
# ---------------------------------------------------------------------------

class _DenseConv(nn.Module):
    """Double-conv block for UNet++ nodes.

    Module indices match checkpoint:
      parent.0.0 = Conv2d (first)
      parent.0.1 = BN     (first)
      parent.1.0 = Conv2d (second)
      parent.1.1 = BN     (second)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.add_module("0", nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ))
        self.add_module("1", nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ))
        self._relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._relu(self._modules["0"](x))
        return self._relu(self._modules["1"](x))


class TrainingUNetPlusPlus(nn.Module):
    """Custom U-Net++ used during training for the SE checkpoint.

    Includes ``self.encoder`` (SMP ResNet-34) so that ``encoder.*``
    checkpoint keys are accepted by strict load.

    Block input channels derived from inspect_checkpoints.py shapes:
      x_0_1: in=128  (upsample enc[1]=64 + enc[0]=64)          -> out=32
      x_1_1: in=192  (upsample enc[2]=128 + enc[1]=64)         -> out=64
      x_2_1: in=384  (upsample enc[3]=256 + enc[2]=128)        -> out=128
      x_3_1: in=768  (upsample enc[4]=512 + enc[3]=256)        -> out=256
      x_0_2: in=160  (upsample dec[1]=64 + enc[0]=64 + x_0_1=32) -> out=32
      x_1_2: in=256  (upsample dec[2]=128 + enc[1]=64 + x_1_1=64) -> out=64
      x_2_2: in=512  (upsample dec[3]=256 + enc[2]=128 + x_2_1=128) -> out=128
      x_0_3: in=192  (upsample dec[1]=64 + enc[0]=64 + 32+32)  -> out=32
      x_1_3: in=320  (upsample dec[2]=128 + enc[1]=64 + 64+64) -> out=64
      x_0_4: in=224  (upsample dec[1]=64 + enc[0]=64 + 32*3)   -> out=32
      final_block: in=32, out=16
      head: (1, 16, 1, 1)
    """

    _ENC = [64, 64, 128, 256, 512]  # ResNet-34 per-stage output channels

    def __init__(self, encoder_name: str = "resnet34", num_classes: int = 1):
        super().__init__()
        self.encoder = _get_encoder(encoder_name)

        enc = self._ENC
        dec = [32, 64, 128, 256]  # decoder output channels per column

        blocks: dict[str, nn.Module] = {}

        # depth-1 nodes
        blocks["x_0_1"] = _DenseConv(enc[1] + enc[0],            dec[0])  # 128->32
        blocks["x_1_1"] = _DenseConv(enc[2] + enc[1],            dec[1])  # 192->64
        blocks["x_2_1"] = _DenseConv(enc[3] + enc[2],            dec[2])  # 384->128
        blocks["x_3_1"] = _DenseConv(enc[4] + enc[3],            dec[3])  # 768->256

        # depth-2 nodes
        blocks["x_0_2"] = _DenseConv(dec[1] + enc[0] + dec[0],  dec[0])  # 160->32
        blocks["x_1_2"] = _DenseConv(dec[2] + enc[1] + dec[1],  dec[1])  # 256->64
        blocks["x_2_2"] = _DenseConv(dec[3] + enc[2] + dec[2],  dec[2])  # 512->128

        # depth-3 nodes
        blocks["x_0_3"] = _DenseConv(dec[1] + enc[0] + dec[0]*2, dec[0])  # 192->32
        blocks["x_1_3"] = _DenseConv(dec[2] + enc[1] + dec[1]*2, dec[1])  # 320->64

        # depth-4 node
        blocks["x_0_4"] = _DenseConv(dec[1] + enc[0] + dec[0]*3, dec[0])  # 224->32

        self.blocks = nn.ModuleDict(blocks)
        self.final_block = _DenseConv(dec[0], 16)   # 32->16
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0, w0 = x.shape[-2], x.shape[-1]
        feats = self.encoder(x)
        x_0_0 = feats[1]  # 64ch, H/2
        x_1_0 = feats[2]  # 64ch, H/4
        x_2_0 = feats[3]  # 128ch, H/8
        x_3_0 = feats[4]  # 256ch, H/16
        x_4_0 = feats[5]  # 512ch, H/32

        def up(feat, target):
            return F.interpolate(feat, size=target.shape[-2:], mode="bilinear", align_corners=False)

        x_0_1 = self.blocks["x_0_1"](torch.cat([up(x_1_0, x_0_0), x_0_0], dim=1))
        x_1_1 = self.blocks["x_1_1"](torch.cat([up(x_2_0, x_1_0), x_1_0], dim=1))
        x_2_1 = self.blocks["x_2_1"](torch.cat([up(x_3_0, x_2_0), x_2_0], dim=1))
        x_3_1 = self.blocks["x_3_1"](torch.cat([up(x_4_0, x_3_0), x_3_0], dim=1))

        x_0_2 = self.blocks["x_0_2"](torch.cat([up(x_1_1, x_0_0), x_0_0, x_0_1], dim=1))
        x_1_2 = self.blocks["x_1_2"](torch.cat([up(x_2_1, x_1_0), x_1_0, x_1_1], dim=1))
        x_2_2 = self.blocks["x_2_2"](torch.cat([up(x_3_1, x_2_0), x_2_0, x_2_1], dim=1))

        x_0_3 = self.blocks["x_0_3"](torch.cat([up(x_1_2, x_0_0), x_0_0, x_0_1, x_0_2], dim=1))
        x_1_3 = self.blocks["x_1_3"](torch.cat([up(x_2_2, x_1_0), x_1_0, x_1_1, x_1_2], dim=1))

        x_0_4 = self.blocks["x_0_4"](torch.cat([up(x_1_3, x_0_0), x_0_0, x_0_1, x_0_2, x_0_3], dim=1))

        x_0_4_up = F.interpolate(x_0_4, size=(h0, w0), mode="bilinear", align_corners=False)
        fb = self.final_block(x_0_4_up)
        return self.head(fb)


# ---------------------------------------------------------------------------
# Builders registered in CUSTOM_BUILDERS
# ---------------------------------------------------------------------------

CUSTOM_BUILDERS["MA"] = lambda meta: TrainingDeepLabV3Plus(encoder_name="resnet50")
CUSTOM_BUILDERS["HE"] = lambda meta: TrainingDeepLabV3Plus(encoder_name="resnet50")
CUSTOM_BUILDERS["EX"] = lambda meta: TrainingUNet(encoder_name="resnet34")
CUSTOM_BUILDERS["SE"] = lambda meta: TrainingUNetPlusPlus(encoder_name="resnet34")
