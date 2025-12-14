import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Data_Gen, Data_Gen_2p5D
from model.network import Net
from utils import DiceScore, IoU, ravd_batch  # assuming these are defined in utils

# --- Paths: match your 2.5D training ---
test_data_path = '/users/PAS3110/sephora20/workspace/PDAC/data/panther/cropped/test/CLARiTY/ten/images_pdac_ktransnn_on_panther_decoder2/'
test_label_path = '/users/PAS3110/sephora20/workspace/PDAC/data/panther/cropped/test/labels/'
test_prompt_file = "/users/PAS3110/sephora20/workspace/PDAC/data/panther/cropped/test/prompts.json"

model_path ='/users/PAS3110/sephora20/workspace/PDAC/LLMs/stage2/outputs/pdac_ktransnn_on_panther/decoder2/ten/weights/best_model.pth'
save_dir   = '/users/PAS3110/sephora20/workspace/PDAC/LLMs/stage2/outputs/pdac_ktransnn_on_panther/decoder2/ten/pred/'
os.makedirs(save_dir, exist_ok=True)

# --- Dataset & DataLoader (2.5D) ---
test_base      = Data_Gen(test_data_path, test_label_path, test_prompt_file, mode='test')
test_dataset   = Data_Gen_2p5D(test_base)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

print(f"Total test samples (2.5D center slices): {len(test_dataset)}")

# --- Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = Net(
    pretrained=True,
    out_channels=2,
    in_channels=6,          # 3 slices × (image + latent)
    use_latent_guidance=True,
    latent_channels=1,
    text_cond=True,
    text_dim=512
).to(device)

state_dict = torch.load(model_path, map_location=device)
network.load_state_dict(state_dict)
network.eval()

# --- Metrics ---
dice_fn = DiceScore(num_classes=2).to(device)
iou_fn  = IoU(num_classes=2).to(device)

dice_total = 0.0
iou_total  = 0.0
ravd_total = 0.0
num_batches = 0
num_images  = 0

dice_list = []
iou_list  = []
ravd_list = []

# --- Evaluation Loop ---
# Track global best Dice
best_dice = -1
best_filename = None

with torch.no_grad():
    for batch_idx, (images, labels, texts, stems, image_filenames) in tqdm(
        enumerate(test_dataloader),
        total=len(test_dataloader),
        desc="Predicting (2.5D)"
    ):
        images = images.to(device)
        labels = labels.to(device)
        texts  = list(texts)

        outputs, _ = network(images, texts=texts)

        dice = dice_fn(outputs, labels)
        dice_val = dice.item()
        dice_total += dice_val
        dice_list.append(dice_val)

        # --- Track highest Dice file ---
        for i in range(len(image_filenames)):
            fname = os.path.basename(image_filenames[i])
            if dice_val > best_dice:
                best_dice = dice_val
                best_filename = fname
        # ---------------------------------

        iou = iou_fn(outputs, labels)
        iou_val = iou.item()
        iou_total += iou_val
        iou_list.append(iou_val)

        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        labels_np = labels.squeeze(1).cpu().numpy()
        preds_np  = preds.cpu().numpy()

        ravd_vals = ravd_batch(labels_np, preds_np)
        ravd_total += np.sum(ravd_vals)
        num_images += preds_np.shape[0]
        ravd_list.extend(ravd_vals.tolist())


# --- Final Results ---
average_dice  = dice_total / num_batches
average_iou   = iou_total / num_batches
average_ravd  = ravd_total / num_images

dice_std  = np.std(dice_list)
iou_std   = np.std(iou_list)
ravd_std  = np.std(ravd_list)

print(f"\n Average Dice coefficient: {average_dice:.4f} ± {dice_std:.4f}")
print(f" Average IoU: {average_iou:.4f} ± {iou_std:.4f}")
print(f" Average RAVD: {average_ravd:.4f} ± {ravd_std:.4f}")
#print(f"Predictions saved to: {save_dir}")
print(f"\nHighest Dice image: {best_filename}  --> Dice = {best_dice:.4f}")

