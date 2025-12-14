import os
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
from PIL import Image
from torch import optim
import torchvision
from torchvision import transforms
import torchvision.models as models
from torchvision.models import vgg16_bn, resnet18, resnet34, resnet50, resnet152, resnet101

from model.blocks import *   # assumes Conv, UpConv, ECA, FAM, Gate, TextEncoder, FiLM, etc. are here


class Net(nn.Module):
    """
    2.5D version of your VGG16-UNet with latent guidance and text conditioning.

    Input x: (B, 6, H, W) with channels:
        0 = img_{i-1}
        1 = lat_{i-1}
        2 = img_i
        3 = lat_i
        4 = img_{i+1}
        5 = lat_{i+1}

    Internally:
        - img_stack = [img_{i-1}, img_i, img_{i+1}]  -> (B,3,H,W)  -> VGG encoder
        - latent_stack = [lat_{i-1}, lat_i, lat_{i+1}] -> (B,3,H,W)
          -> lat = mean(latent_stack, dim=1, keepdim=True) -> (B,1,H,W)
        - 'lat' is injected at each encoder scale via lat1..latb
    """

    def __init__(self,
                 pretrained=True,
                 out_channels=3,
                 in_channels=6,          # 2.5D: 3 slices × (img+latent) = 6
                 use_latent_guidance=True,
                 latent_channels=1,      # we aggregate latent_stack -> 1 channel
                 text_cond=True,
                 text_dim=512):
        super().__init__()
        self.use_latent_guidance = use_latent_guidance
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.text_cond = text_cond

        # 1) Input stem: take 3 image channels (2.5D stack) -> 3 channels for VGG
        #    We no longer project arbitrary C_in; we explicitly use 3 image channels.
        self.input_stem = nn.Conv2d(3, 3, kernel_size=1, bias=False)

        # ------- VGG16-BN encoder & your attention blocks -------
        self.encoder = vgg16_bn(pretrained=pretrained).features

        self.block1 = nn.Sequential(*self.encoder[:6])
        self.eca1 = ECA(64)
        self.Att1 = FAM(F_g=64, F_l=64, F_int=32)

        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.eca2 = ECA(128)
        self.Att2 = FAM(F_g=128, F_l=128, F_int=64)

        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.eca3 = ECA(256)
        self.Att3 = FAM(F_g=256, F_l=256, F_int=128)

        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.eca4 = ECA(512)
        self.Att4 = FAM(F_g=512, F_l=512, F_int=256)

        self.block5 = nn.Sequential(*self.encoder[27:34])
        self.eca5 = ECA(512)
        self.Att5 = FAM(F_g=512, F_l=512, F_int=256)

        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.ecab = ECA(512)
        self.Attb = FAM(F_g=512, F_l=512, F_int=256)
        self.conv_bottleneck = Conv(512, 1024)

        self.up_conv6 = UpConv(1024, 512)
        self.Att6 = FAM(F_g=512, F_l=512, F_int=256)
        self.conv6 = Conv(512 + 512, 512)

        self.up_conv7 = UpConv(512, 256)
        self.Att7 = FAM(F_g=256, F_l=256, F_int=128)
        self.conv7 = Conv(256 + 512, 256)

        self.up_conv8 = UpConv(256, 128)
        self.Att8 = FAM(F_g=128, F_l=128, F_int=64)
        self.conv8 = Conv(128 + 256, 128)

        self.up_conv9 = UpConv(128, 64)
        self.Att9 = FAM(F_g=64, F_l=64, F_int=32)
        self.conv9 = Conv(64 + 128, 64)

        self.up_conv10 = UpConv(64, 32)
        self.conv10 = Conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)

        self.gate = Gate(dims=64, head=1)

        # 2) tiny convs to inject latent at each scale
        if self.use_latent_guidance:
            self.lat1 = nn.Conv2d(latent_channels, 64, 1)
            self.lat2 = nn.Conv2d(latent_channels, 128, 1)
            self.lat3 = nn.Conv2d(latent_channels, 256, 1)
            self.lat4 = nn.Conv2d(latent_channels, 512, 1)
            self.lat5 = nn.Conv2d(latent_channels, 512, 1)
            self.latb = nn.Conv2d(latent_channels, 512, 1)  # bottleneck

        # 3) Text conditioning (FiLM), as you had
        if self.text_cond:
            self.txt = TextEncoder(model='clip', text_dim=text_dim, freeze=True)
            # FiLM per scale
            self.f1 = FiLM(text_dim, 64)
            self.f2 = FiLM(text_dim, 128)
            self.f3 = FiLM(text_dim, 256)
            self.f4 = FiLM(text_dim, 512)
            self.f5 = FiLM(text_dim, 512)
            self.fb = FiLM(text_dim, 512)
            # tiny gate to control strength
            self.alpha = nn.Parameter(torch.zeros(6))  # [f1,f2,f3,f4,f5,fb]

    def _split_image_latent(self, x):
        """
        2.5D split.

        x: (B, 6, H, W) with channels:
          0 = img_{i-1}
          1 = lat_{i-1}
          2 = img_i
          3 = lat_i
          4 = img_{i+1}
          5 = lat_{i+1}

        Returns:
          img_stack: (B,3,H,W) = [img_{i-1}, img_i, img_{i+1}]
          lat:       (B,1,H,W) = mean([lat_{i-1}, lat_i, lat_{i+1}])
        """
        B, C, H, W = x.shape
        assert C == 6, f"2.5D Net expects 6 channels, got {C}"

        # Extract the 3 image slices
        img_stack = torch.stack(
            [x[:, 0, :, :], x[:, 2, :, :], x[:, 4, :, :]],
            dim=1
        )  # (B,3,H,W)

        # Extract the 3 latent slices
        latent_stack = torch.stack(
            [x[:, 1, :, :], x[:, 3, :, :], x[:, 5, :, :]],
            dim=1
        )  # (B,3,H,W)

        # Aggregate 3 latents → 1 latent map
        lat = latent_stack.mean(dim=1, keepdim=True)  # (B,1,H,W)

        return img_stack, lat

    def forward(self, x, texts):
        """
        x: (B, 6, H, W)  = 2.5D [img, latent] for 3 slices
        texts: list[str] length B
        """
        # 2.5D split
        img_stack, lat = self._split_image_latent(x)   # img_stack: (B,3,H,W), lat: (B,1,H,W)

        # Project 3 image channels to 3 for VGG
        x3 = self.input_stem(img_stack)  # (B,3,H,W)

        if self.text_cond and texts is not None:
            dev = x3.device
            t = self.txt.encode(texts, device=dev)
        else:
            t = None

        # ---------- Encoder ----------
        block1 = self.block1(x3)
        if self.use_latent_guidance and lat is not None:
            l1 = F.interpolate(lat, size=block1.shape[-2:], mode='bilinear', align_corners=False)
            block1 = block1 + self.lat1(l1)
        if t is not None:
            block1 = block1 + self.alpha[0] * (self.f1(block1, t) - block1)
        block1 = self.eca1(block1)
        block1, p1 = self.Att1(g=block1, x=block1)

        block2 = self.block2(block1)
        if self.use_latent_guidance and lat is not None:
            l2 = F.interpolate(lat, size=block2.shape[-2:], mode='bilinear', align_corners=False)
            block2 = block2 + self.lat2(l2)
        if t is not None:
            block2 = block2 + self.alpha[1] * (self.f2(block2, t) - block2)
        block2 = self.eca2(block2)
        block2, p2 = self.Att2(g=block2, x=block2)

        block3 = self.block3(block2)
        if self.use_latent_guidance and lat is not None:
            l3 = F.interpolate(lat, size=block3.shape[-2:], mode='bilinear', align_corners=False)
            block3 = block3 + self.lat3(l3)
        if t is not None:
            block3 = block3 + self.alpha[2] * (self.f3(block3, t) - block3)
        block3 = self.eca3(block3)
        block3, p3 = self.Att3(g=block3, x=block3)

        block4_i = self.block4(block3)
        if self.use_latent_guidance and lat is not None:
            l4 = F.interpolate(lat, size=block4_i.shape[-2:], mode='bilinear', align_corners=False)
            block4_i = block4_i + self.lat4(l4)
        block4 = self.eca4(block4_i)
        if t is not None:
            block4 = block4 + self.alpha[3] * (self.f4(block4, t) - block4)
        block4, p4 = self.Att4(g=block4, x=block4)

        block5_i = self.block5(block4)
        if self.use_latent_guidance and lat is not None:
            l5 = F.interpolate(lat, size=block5_i.shape[-2:], mode='bilinear', align_corners=False)
            block5_i = block5_i + self.lat5(l5)
        block5 = self.eca5(block5_i)
        if t is not None:
            block5 = block5 + self.alpha[4] * (self.f5(block5, t) - block5)
        block5, p5 = self.Att5(g=block5, x=block5)

        bottleneck = self.bottleneck(block5)
        if self.use_latent_guidance and lat is not None:
            lb = F.interpolate(lat, size=bottleneck.shape[-2:], mode='bilinear', align_corners=False)
            bottleneck = bottleneck + self.latb(lb)
        bottleneck = self.ecab(bottleneck)
        if t is not None:
            bottleneck = bottleneck + self.alpha[5] * (self.fb(bottleneck, t) - bottleneck)
        bottleneck, pb = self.Attb(g=bottleneck, x=bottleneck)

        x_dec = self.conv_bottleneck(bottleneck)

        # Bridge
        block4, block5 = self.gate([block4_i, block5_i])

        # ---------- Decoder ----------
        x_dec = self.up_conv6(x_dec)
        x_dec, u6 = self.Att6(g=x_dec, x=x_dec)
        x_dec = torch.cat([x_dec, block5], dim=1)
        x_dec = self.conv6(x_dec)

        x_dec = self.up_conv7(x_dec)
        x_dec, u7 = self.Att7(g=x_dec, x=x_dec)
        x_dec = torch.cat([x_dec, block4], dim=1)
        x_dec = self.conv7(x_dec)

        x_dec = self.up_conv8(x_dec)
        x_dec, u8 = self.Att8(g=x_dec, x=x_dec)
        x_dec = torch.cat([x_dec, block3], dim=1)
        x_dec = self.conv8(x_dec)

        x_dec = self.up_conv9(x_dec)
        x_dec, u9 = self.Att9(g=x_dec, x=x_dec)
        x_dec = torch.cat([x_dec, block2], dim=1)
        x_dec = self.conv9(x_dec)

        x_dec = self.up_conv10(x_dec)
        x_dec = torch.cat([x_dec, block1], dim=1)
        x_dec = self.conv10(x_dec)
        x_dec = self.conv11(x_dec)

        return x_dec, [p1, p2, p3, p4, p5, pb, u6, u7, u8, u9]
