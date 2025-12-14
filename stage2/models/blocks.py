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
from model.helpers import *

        
class Gate(nn.Module):
    def __init__(self, dims, head, ratio=None):
        super().__init__()
        self.attn1_block = CSAB(dims, head, ratio)
        self.attn2_block = CSAB(dims, head, ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x[0].shape[0]
        C = 64
        c4, c5 = x

        c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
        c5f = c5.permute(0, 2, 3, 1).reshape(B, -1, C)
        # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
        res_x = torch.cat([c4f, c5f], -2)        

        attn1 = self.attn1_block(x) + res_x
        attn2 = self.attn2_block(attn1) + attn1

        B,_,C = attn2.shape
        outs = []
        
        sk4 = attn2[:,:6272,:].reshape(B, 28, 28, C*8).permute(0, 3, 1, 2)
        sk5 = attn2[:,6272:,:].reshape(B, 14, 14, C*8).permute(0, 3, 1, 2)

        outs.append(sk4)
        outs.append(sk5)
        
        return outs  

class FAM(nn.Module): #Feature Attention Module
    def __init__(self,F_g,F_l,F_int):
        super(FAM,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi, psi
        
class CSAB(nn.Module):  # CrossScaleAttentionBlock
    def __init__(self, dims, head, ratios=None):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = TASA(dims, head, ratio=None)
        self.norm2 = nn.LayerNorm(dims)
        

        self.mixffn4 = SkipConv(dims*8,dims*32)
        self.mixffn5 = SkipConv(dims*8,dims*32)
        
    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c4, c5 = inputs
  
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
            c5f = c5.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64

            inputs = torch.cat([c4f, c5f], -2)

        else:
            B,_,C = inputs.shape 

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        tem4 = tx[:,:6272,:].reshape(B, -1, C*8)

        tem5 = tx[:,6272:,:].reshape(B, -1, C*8)


        m4f = self.mixffn4(tem4, 28, 28).reshape(B, -1, C)
        m5f = self.mixffn4(tem5, 14, 14).reshape(B, -1, C)


        t1 = torch.cat([m4f, m5f], -2)

        
        tx2 = tx1 + t1

        return tx2
        
class TASA(nn.Module): # TokenAwareSelfAttention
    def __init__(self, dim, head, ratio):
        super().__init__()
        self.head = head
        self.ratio = ratio 
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim*2, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        if ratio is not None:
            self.fusion = MultiScaleFusion(dim,ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.ratio is not None:
            x = self.fusion(x)
            
        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)
        return out
        
