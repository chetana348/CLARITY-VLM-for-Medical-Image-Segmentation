import os
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler,RandomSampler
from torch.cuda import amp
from PIL import Image
from torch import nn
from torch import optim
import torchvision
from torchvision import transforms
import torchvision.models as models
from torchvision.models import vgg16_bn, resnet18, resnet34, resnet50, resnet152, resnet101
import clip


def UpConv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )
    
def Conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    
class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)+x 


# ---- FiLM (feature-wise linear modulation) ----
class FiLM(nn.Module):
    def __init__(self, text_dim: int, feat_dim: int):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(text_dim, 2 * feat_dim),
        )
    def forward(self, x, t):
        # x: (B,C,H,W), t: (B,text_dim)
        gamma_beta = self.lin(t)                  # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=1)  # (B,C) each
        gamma = gamma.unsqueeze(-1).unsqueeze(-1) # (B,C,1,1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta

# ---- Text encoder wrapper ----
class TextEncoder(nn.Module):
    def __init__(self, model='clip', text_dim=512, freeze=True, vocab_size=30522, emb_dim=256, hidden=512):
        super().__init__()
        self.kind = model
        self.text_dim = text_dim
        if model == 'clip':
            # Requires: pip install openai-clip  (or use transformers' CLIPTextModel)
            import clip
            self.clip_model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
            if freeze:
                for p in self.clip_model.parameters():
                    p.requires_grad = False
            self.register_buffer("_dummy", torch.empty(0))  # allows .to(device) on the module
        else:
            # lightweight fallback: Embedding + GRU → pooled vector
            self.emb = nn.Embedding(vocab_size, emb_dim)
            self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
            self.proj = nn.Linear(2*hidden, text_dim)

    def encode(self, texts, device):
        """
        texts: list[str] length B
        returns (B, text_dim)
        """
        if next(self.clip_model.parameters()).device != device:
            self.clip_model = self.clip_model.to(device)
        # tokenize -> move tokens to SAME device as CLIP
        tokens = clip.tokenize(texts).to(device)              # (B,77), long on device
        z = self.clip_model.encode_text(tokens)               # (B,512), possibly fp16 on CUDA
        z = z.float()                                         # make fp32
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)         # L2 normalize
        return z     
        
class SkipConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DepthConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)
    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W)+self.fc1(x)))
        out = self.fc2(ax)
        return out
        
class DepthConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.conv(tx)
        return conv_x.flatten(2).transpose(1, 2)
        
class MultiScaleFusion(nn.Module):
    def __init__(self, dim, ratio=[1, 2, 4, 8, 16]):
        super().__init__()
        self.dim = dim
        self.ratio = ratio

        self.sr0 = nn.Conv2d(dim, dim, ratio[4], ratio[4])
        self.sr1 = nn.Conv2d(dim*2, dim*2, ratio[3], ratio[3])
        self.sr2 = nn.Conv2d(dim*4, dim*4, ratio[2], ratio[2])
        self.sr3 = nn.Conv2d(dim*8, dim*8, ratio[1], ratio[1])
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        tem0 = x[:,:50176,:].reshape(B, 224, 224, C).permute(0, 3, 1, 2) 
        tem1 = x[:,50176:75264,:].reshape(B, 112, 112, C*2).permute(0, 3, 1, 2)
        tem2 = x[:,75264:87808,:].reshape(B, 56, 56, C*4).permute(0, 3, 1, 2)
        tem3 = x[:,87808:94080,:].reshape(B, 28, 28, C*8).permute(0, 3, 1, 2)
        tem4 = x[:,94080:95648,:]

        sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
        sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
        sr_2 = self.sr2(tem2).reshape(B, C, -1).permute(0, 2, 1)
        sr_3 = self.sr3(tem3).reshape(B, C, -1).permute(0, 2, 1)


        reduce_out = self.norm(torch.cat([sr_0, sr_1, sr_2, sr_3, tem4], -2))
        
        return reduce_out