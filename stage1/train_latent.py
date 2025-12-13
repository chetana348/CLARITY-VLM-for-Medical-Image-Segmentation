# %%

seed = 42
roi_size = (128, 128)    # 2D images: 128x128 (not really used now, but harmless)
import warnings
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
import os
import time
import sys
sys.path.append(os.path.dirname(__file__))
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import multiprocessing
import monai
from monai.transforms import (
    AsDiscrete, Compose, Activations
)
import torch
from monai.data import DataLoader, decollate_batch
from monai.utils import first
num_workers = 1

import random
from torch.utils.data import Subset

from monai.networks.layers import Norm
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from models.losses import *
from dataset import *
from pathlib import Path
from models.clinvlm import *
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

dataset_dir = './outputs/pros_x/weights_2classes/'
os.makedirs(dataset_dir, exist_ok=True)
device = torch.device("cuda:0")

import torch.nn.functional as F

desc_csv = "/users/PAS3110/sephora20/workspace/PDAC/LLMs/stage1/texts/pros_mri_descriptions.csv"
train, val, test, class_definitions = DataGen(
    seed,
    desc_csv=desc_csv,
)

# For single-class (tumor-only) training, background is implicit (0), tumor is 1
classes_dict = {
    'PZ': 1,
    'TZ': 2
}


# ---- single output channel: latent / tumor logit ----
num_channels = 2
val_subset  = Subset(val,  list(range(min(40, len(val)))))
test_subset = Subset(test, list(range(min(40, len(test)))))

# %%
train_loader = DataLoader(train, batch_size=4, shuffle=True,  num_workers=num_workers)
#val_loader   = DataLoader(val,   batch_size=4, shuffle=False, num_workers=num_workers)
val_loader  = DataLoader(val_subset,  batch_size=4, shuffle=False, num_workers=num_workers)

# %%
max_epochs = 300
val_interval = 1
VAL_AMP = True

# %%
from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- Single-channel DiceCE with sigmoid (no softmax) ----
seg_loss = monai.losses.DiceCELoss(
    smooth_nr=0,
    smooth_dr=1e-5,
    squared_pred=True,
    to_onehot_y=False,   # labels are [B,1,H,W] with 0/1
    sigmoid=True,        # single-channel logits + sigmoid
)

# Define the Linear Projection Layer
projection_dim = 768  # This should match the embedding dimension of text_embeddings

cp_feats_visual = nn.Linear(in_features=1, out_features=128, device=device)
cp_feats_text   = nn.Linear(in_features=projection_dim, out_features=128, device=device)

# infer in_channels from dataset instead of hardcoding
sample = train[0]
in_channels = sample["image"].shape[0]  # e.g. 1 for grayscale

model = ClinicalVLM(
            img_size=(128,128),
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=num_channels,
            feature_size=48,
            device=device,
            text_encoder="pubmedbert",      #bert, biobert, clinicalbert, pubmedbert
            use_checkpoint=True,
            use_v2 = True,
        ).to(device)


# Initialize the optimizer with all the parameters, including from the enhanced embedding module
params = list(model.parameters()) + list(cp_feats_visual.parameters())
if cp_feats_text is not None:
    params += list(cp_feats_text.parameters())

optimizer = torch.optim.AdamW(
    params,
    lr=1e-4,
    weight_decay=1e-5
)

# Dice metric on binarized predictions (for validation only)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

# %%
# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()

# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

best_metric = -1
best_metric_epoch = -1

# ---- For validation: sigmoid + threshold = binary mask ----
post_pred = Compose([
    Activations(sigmoid=True),
    AsDiscrete(threshold=0.5),
])

# %%
total_start = time.time()

from tqdm import tqdm

# Define learnable task uncertainties
log_sigma1 = torch.nn.Parameter(torch.tensor(0.0, device=device))  # Segmentation/latent loss uncertainty
log_sigma2 = torch.nn.Parameter(torch.tensor(0.0, device=device))  # Feature consistency loss uncertainty
log_sigma3 = torch.nn.Parameter(torch.tensor(0.0, device=device))  # InfoNCE loss uncertainty

# Register parameters in optimizer
optimizer.add_param_group({'params': [log_sigma1, log_sigma2, log_sigma3]})

# ===========================
#          TRAINING
# ===========================
for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10)
    model.train()
    epoch_loss = 0.0
    epoch_loss1 = 0.0
    epoch_loss2 = 0.0
    epoch_loss3 = 0.0
    epoch_loss4 = 0.0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),  # [B, C, H, W]
            batch_data["label"].to(device),  # [B, 1, H, W] with 0/1
        )

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda"):

            # grab per-image long descriptions from the batch (list of len B)
            image_descriptions = batch_data["description"]  # e.g. ["This slice shows ...", ...]
        
            # forward with per-image descriptions
            feats, class_text_feats, logit_map, img_text_emb = model(
                x=inputs,
                class_definitions=class_definitions,   # or None if you want to drop class-level text
                image_descriptions=image_descriptions,
                device=device,
            )
        
            # choose what you treat as "visual_embeddings" for the contrastive losses
            # here feats is [B, 1] because out_channels=num_channels=1
            visual_embeddings = feats              # [B, 1]
            text_embeddings = img_text_emb         # [B, L, D]
        
            # ---- 2D feature/logit alignment, unchanged ----
            B = inputs.shape[0]
            h, w = visual_embeddings.shape[-2:] if visual_embeddings.ndim == 4 else labels.shape[-2:]
        
            # downsample logits to feature resolution (2D)
            logits_downsampled = F.interpolate(
                logit_map, size=(h, w), mode='bilinear', align_corners=False
            )
        
            # If you still want CE_loss on a spatial map, use a spatial feature map instead of feats.
            # For now, we keep your original shapes (1 channel, HW flattened):
        
            visual_map = logit_map                 # [B, 1, H, W]
            visual_map_down = logits_downsampled   # [B, 1, h, w]
        
            #visual_map   = visual_map.view(B, num_channels, -1)
            #visual_down  = visual_map_down.view(B, num_channels, -1)
        
            # ---- Latent loss (DiceCE / segmentation) ----
            loss1 = seg_loss(logit_map, labels)
        
            # Feature consistency loss (loss2)
            loss2 = l1_loss(visual_map, visual_map_down)
        
            # InfoNCE loss (loss3) to align feats <-> text
            loss3 = info_nce_loss(
                visual_embeddings,   # [B, 1]
                text_embeddings,     # [B, L, D]
                cp_feats_visual,
                cp_feats_text,
                device,
                0.2,
            )
        
            # Text–image alignment loss (loss4)
            loss4 = support_loss(
                visual_embeddings,   # [B, 1]
                text_embeddings,     # [B, L, D]
                cp_feats_visual,
                cp_feats_text,
                device,
            )
        
            lambda_ti = 0.1
        
            # Total loss with homoscedastic uncertainty
            loss = (
                (loss1 / (2 * torch.exp(log_sigma1))) + log_sigma1 +
                (loss2 / (2 * torch.exp(log_sigma2))) + log_sigma2 +
                (loss3 / (2 * torch.exp(log_sigma3))) + log_sigma3 +
                lambda_ti * loss4
            )
        
        
            # Backpropagation
            scaler.scale(loss).backward()
    
            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()
            epoch_loss3 += loss3 if isinstance(loss3, float) else loss3.item()
            epoch_loss4 += loss4.item()
            epoch_loss  += loss.item()
    
            # Optimizer steps
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
    
            # (lr_scheduler can be re-added if you want)
    
        # After the epoch ends
        epoch_loss  /= step
        epoch_loss1 /= step
        epoch_loss2 /= step
        epoch_loss3 /= step
        epoch_loss4 /= step

    print(
        f"Epoch {epoch + 1}, "
        f"Average Loss1 (DiceCE/latent): {epoch_loss1:.4f}, "
        f"Average Loss2 (CE): {epoch_loss2:.4f}, "
        f"Average Loss3 (InfoNCE): {epoch_loss3:.4f}, "
        f"Average Loss4 (InfoNCE): {epoch_loss4:.4f}, "
        f"Average Total Loss: {epoch_loss:.4f}"
    )

    # ===========================
    #        VALIDATION
    # ===========================
    if (epoch + 5) % val_interval == 0:
        model.eval()
        with torch.inference_mode():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )

                # Direct forward pass (no sliding window)
                _, _, val_logits, _ = model(
                    x=val_inputs,
                    class_definitions=class_definitions,
                    image_descriptions=None,  # no text used at validation, or you could also pass batch descriptions
                    device=device,
                )

                # For Dice: binarize with sigmoid + threshold
                val_outputs = [post_pred(i) for i in decollate_batch(val_logits)]
                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save({
                    'model': model.state_dict(),
                    'cp_feats_visual': cp_feats_visual.state_dict(),
                    'cp_feats_text': cp_feats_text.state_dict() if cp_feats_text is not None else None,
                    'log_sigma1': log_sigma1.detach().cpu(),
                    'log_sigma2': log_sigma2.detach().cpu(),
                    'log_sigma3': log_sigma3.detach().cpu(),
                }, os.path.join(dataset_dir, "model.pth"))
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")

total_time = time.time() - total_start
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
