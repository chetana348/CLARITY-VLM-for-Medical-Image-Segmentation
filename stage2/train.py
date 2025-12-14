import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

from dataset import *              # your existing Data_Gen
from model.network import Net      # make sure this is the 2.5D Net you just defined
from utils import *                # DiceLoss, DiceScore, IoU, etc.



# -----------------------------
# Paths
# -----------------------------
train_data_path = '/users/PAS3110/sephora20/workspace/PDAC/data/panther/cropped/train/CLARiTY/ten/images_pdac_ktransnn_on_panther_decoder2/'
train_label_path = '/users/PAS3110/sephora20/workspace/PDAC/data/panther/cropped/train/labels/'
train_prompt_file = "/users/PAS3110/sephora20/workspace/PDAC/data/panther/cropped/train/prompts.json"

test_data_path = '/users/PAS3110/sephora20/workspace/PDAC/data/panther/cropped/test/CLARiTY/ten/images_pdac_ktransnn_on_panther_decoder2/'
test_label_path = '/users/PAS3110/sephora20/workspace/PDAC/data/panther/cropped/test/labels/'
test_prompt_file = "/users/PAS3110/sephora20/workspace/PDAC/data/panther/cropped/test/prompts.json"

save_dir = '/users/PAS3110/sephora20/workspace/PDAC/LLMs/stage2/outputs/pdac_ktransnn_on_panther/decoder2/ten/weights/'
model_path = '/users/PAS3110/sephora20/workspace/PDAC/LLMs/stage2/outputs/pdac_ktrans/decoder2/weights/best_model.pth'
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# Base per-slice datasets
# -----------------------------
train_base = Data_Gen(train_data_path, train_label_path, train_prompt_file, mode='train')
test_base  = Data_Gen(test_data_path,  test_label_path,  test_prompt_file,  mode='test')

# -----------------------------
# 2.5D datasets & loaders
# -----------------------------
train_dataset = Data_Gen_2p5D(train_base)
test_dataset  = Data_Gen_2p5D(test_base)
train_subset  = Subset(train_dataset, list(range(min(10, len(test_dataset)))))
test_subset = Subset(test_dataset, list(range(min(40, len(test_dataset)))))

train_loader = DataLoader(train_subset, batch_size=2, shuffle=True)
test_loader  = DataLoader(test_subset,  batch_size=2, shuffle=False)


# -----------------------------
# Model: 2.5D Net
# -----------------------------
network = Net(
    pretrained=True,
    out_channels=2,
    in_channels=6,          # 2.5D: 3 slices × (img+latent)
    use_latent_guidance=True,
    latent_channels=1,      # we aggregate 3 latents → 1 inside Net
    text_cond=True,
    text_dim=512
).cuda()

state_dict = torch.load(model_path, map_location='cuda')
network.load_state_dict(state_dict)
#network.eval()

# -----------------------------
# Loss, metrics, optimizer
# -----------------------------
criterion = DiceLoss(num_classes=2)
accuracy_metric = DiceScore(num_classes=2)
iou_metric = IoU(num_classes=2)

optimizer = torch.optim.Adam(network.parameters(), lr=0.0001, betas=(0.9, 0.999))


num_epochs = 1000
best_accuracy = 0.0


# -----------------------------
# Training loop
# -----------------------------
for epoch in range(num_epochs):
    network.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for batch_idx, (images, labels, texts, stems, img_name) in pbar:
        images = images.cuda()   # (B,6,H,W)
        labels = labels.cuda()
        texts  = list(texts)
 
        optimizer.zero_grad()

        # forward: Net returns (seg_logits, aux)
        seg_logits, aux = network(images, texts=texts)   # seg_logits: (B,2,H,W)
        #print(torch.unique(torch.argmax(seg_logits.softmax(1), 1)))
        loss = criterion(seg_logits, labels)
        loss.backward()
        optimizer.step()

        dice = accuracy_metric(seg_logits, labels)
        iou  = iou_metric(seg_logits, labels)

        total_loss += loss.item()
        total_dice += dice.item()
        total_iou  += iou.item()

        pbar.set_postfix(loss=loss.item(), dice=dice.item(), iou=iou.item())

    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    avg_iou  = total_iou  / len(train_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'  Train -> Loss: {avg_loss:.4f}  Dice: {avg_dice:.4f}  IoU: {avg_iou:.4f}')

    # -----------------------------
    # Validation
    # -----------------------------
    network.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou  = 0.0

    pbar_val = tqdm(enumerate(test_loader), total=len(test_loader),
                    desc="Validation", leave=False)

    with torch.no_grad():
        for batch_idx, (images_test, labels_test, texts, stems, img_name) in pbar_val:
            images_test = images_test.cuda()
            labels_test = labels_test.cuda()
            texts = list(texts)

            seg_logits_val, aux_val = network(images_test, texts=texts)

            loss_val = criterion(seg_logits_val, labels_test)
            dice_val = accuracy_metric(seg_logits_val, labels_test)
            iou_val  = iou_metric(seg_logits_val, labels_test)

            val_loss += loss_val.item()
            val_dice += dice_val.item()
            val_iou  += iou_val.item()

            pbar_val.set_postfix(loss=loss_val.item(), dice=dice_val.item(), iou=iou_val.item())

    avg_val_loss = val_loss / len(test_loader)
    avg_val_dice = val_dice / len(test_loader)
    avg_val_iou  = val_iou  / len(test_loader)

    print(f'  Val   -> Loss: {avg_val_loss:.4f}  Dice: {avg_val_dice:.4f}  IoU: {avg_val_iou:.4f}')

    # save best based on validation Dice
    if avg_val_dice > best_accuracy:
        best_accuracy = avg_val_dice
        best_model_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(network.state_dict(), best_model_path)
        print(f"  >> New best model saved with Dice {avg_val_dice:.4f}")

print("Training finished.")
