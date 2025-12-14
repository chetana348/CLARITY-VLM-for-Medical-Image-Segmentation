import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from torch.nn import functional as F
from torchvision import transforms
import time
from torch.utils.data import Dataset, DataLoader

import os
import glob
import json
import torch
import random
import torch.nn as nn
import numpy as np
import torch.utils.data
from torchvision import transforms
import torch.utils.data as data
import torch.nn.functional as F
import cv2
import sys
import albumentations as A
from sklearn.model_selection import KFold

def ravd_batch(y_true_batch, y_pred_batch, epsilon=1e-6):
    batch_ravd = []
    for y_true, y_pred in zip(y_true_batch, y_pred_batch):
        gt_area = np.sum(y_true == 1)
        pred_area = np.sum(y_pred == 1)
        if gt_area == 0:
            ravd = 0.0 if pred_area == 0 else 1.0  # handle empty ground truth
        else:
            ravd = abs(pred_area - gt_area) / (gt_area + epsilon)
        batch_ravd.append(ravd)
    return np.array(batch_ravd)

class DiceLoss(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = 1.

    def forward(self, pred, target):
        target = target.squeeze(1)
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            loss = 1 -  (2. * intersection + self.smooth) / (union + self.smooth)
        else:
            pred = F.softmax(pred, dim=1)
            loss = 0
            for c in range(self.num_classes):
                pred_c = pred[:, c, :, :]
                target_c = (target == c).float()
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                loss += 1 -  (2. * intersection + self.smooth) / (union + self.smooth)
            loss /= self.num_classes
        return loss


class DiceScore(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = 1.

    def forward(self, pred, target):
        target = target.squeeze(1).float()
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
        else:
            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            pred = F.one_hot(pred, self.num_classes).permute(0, 3, 1, 2).float()
            dice = 0
            for c in range(self.num_classes):
                pred_c = pred[:, c, :, :]
                target_c = (target == c).float()
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                dice += (2. * intersection + self.smooth) / (union + self.smooth)
            dice /= self.num_classes
        return dice
    
    
class IoU(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = 1.

    def forward(self, pred, target):
        target = target.squeeze(1).float()
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            den = union - intersection
            iou = (intersection + self.smooth) / (den + self.smooth)
        else:
            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            pred = F.one_hot(pred, self.num_classes).permute(0, 3, 1, 2).float()
            iou = 0
            for c in range(self.num_classes):
                pred_c = pred[:, c, :, :]
                target_c = (target == c).float()
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                den = union - intersection
                iou += (intersection + self.smooth) / (den + self.smooth)
            iou /= self.num_classes
        return iou