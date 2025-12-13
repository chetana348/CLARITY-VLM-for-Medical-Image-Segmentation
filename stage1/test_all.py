import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_grab import pdac
from model_grab import def_model

import tifffile as tiff  # for saving latents as .tif


# ===========================
#     METRIC DEFINITIONS
# ===========================

def ravd_batch(y_true_batch, y_pred_batch, epsilon=1e-6):
    """
    Compute Relative Absolute Volume Difference (RAVD) for a batch.
    y_true_batch, y_pred_batch: numpy arrays of shape [B, H, W] with labels {0,1}.
    """
    batch_ravd = []
    for y_true, y_pred in zip(y_true_batch, y_pred_batch):
        gt_area   = np.sum(y_true == 1)
        pred_area = np.sum(y_pred == 1)
        if gt_area == 0:
            # If GT is empty: RAVD = 0 if pred is also empty, else 1
            ravd = 0.0 if pred_area == 0 else 1.0
        else:
            ravd = abs(pred_area - gt_area) / (gt_area + epsilon)
        batch_ravd.append(ravd)
    return np.array(batch_ravd)


class DiceScore(nn.Module):
    """
    Binary Dice (num_classes=1) or multi-class Dice (num_classes>1).
    For your PDAC case, we use num_classes=1.
    """
    def __init__(self, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = 1.

    def forward(self, pred, target):
        """
        pred: logits, shape [B, C, H, W]
        target: labels, shape [B, 1, H, W] with values in {0,1} (for binary)
        """
        target = target.squeeze(1).float()  # [B, H, W]

        if self.num_classes == 1:
            # Binary case: sigmoid + threshold
            pred = torch.sigmoid(pred)
            pred = (pred >= 0.5).float()
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
        else:
            # Multi-class (not used here)
            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)            # [B, H, W]
            pred = F.one_hot(pred, self.num_classes)    # [B, H, W, C]
            pred = pred.permute(0, 3, 1, 2).float()     # [B, C, H, W]

            dice = 0
            for c in range(self.num_classes):
                pred_c   = pred[:, c, :, :]
                target_c = (target == c).float()
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                dice += (2. * intersection + self.smooth) / (union + self.smooth)
            dice /= self.num_classes

        return dice


# ===========================
#   FILENAME HELPER
# ===========================

def get_paths_from_batch(batch_data, B):
    """
    Try to recover original image paths / names from a MONAI-style batch_data.
    Returns a list of length B, possibly containing None where no path is found.
    """
    paths = [None] * B

    # ---- 1) MONAI-style meta dict ----
    if "image_meta_dict" in batch_data:
        meta = batch_data["image_meta_dict"]

        # Case A: collated into a dict of lists: meta["filename_or_obj"][b]
        if isinstance(meta, dict) and "filename_or_obj" in meta:
            raw = meta["filename_or_obj"]
            if isinstance(raw, (list, tuple)):
                for i in range(min(B, len(raw))):
                    paths[i] = raw[i]
            else:
                # same value for all if not a list
                paths = [raw] * B

        # Case B: list/tuple of per-sample dicts: meta[b]["filename_or_obj"]
        elif isinstance(meta, (list, tuple)):
            tmp = []
            for m in meta:
                if isinstance(m, dict) and "filename_or_obj" in m:
                    tmp.append(m["filename_or_obj"])
                else:
                    tmp.append(None)
            if any(p is not None for p in tmp):
                for i in range(min(B, len(tmp))):
                    paths[i] = tmp[i]

    # ---- 2) Other common keys in your own Dataset ----
    # Adjust this list if your dataset uses a different key name
    for key in ["image_path", "img_path", "path", "name", "id"]:
        if key in batch_data:
            raw = batch_data[key]
            if isinstance(raw, (list, tuple)):
                for i in range(min(B, len(raw))):
                    paths[i] = raw[i]
            else:
                paths = [raw] * B
            break

    return paths

def main():
    # ---- Config ----
    seed        = 42
    dataset_dir = "./outputs/test/latents"   # same as in training
    batch_size  = 1

    # Base dir to hold all latent folders
    latents_root = os.path.join(dataset_dir, "latents_all")
    os.makedirs(latents_root, exist_ok=True)

    latent_module_names = [
        "encoder1",
        "encoder4",
        "encoder10",
        "decoder2",
        "decoder1",
    ]

    latent_dirs = {
        name: os.path.join(latents_root, name)
        for name in latent_module_names + ["softmask"]
    }
    for d in latent_dirs.values():
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ===========================
    #       LOAD DATASET
    # ===========================
    train_ds, val_ds, test_ds, class_definitions = pdac(seed)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    # ===========================
    #       BUILD THE MODEL
    # ===========================
    in_channels = 1
    num_channels = 2

    model = def_model(
        model_type,
        in_channels,
        num_channels,
        device,
        text_encoder="pubmedbert",
    )
    model.to(device)

    # ===========================
    #       LOAD CHECKPOINT
    # ===========================
    ckpt_path = os.path.join(dataset_dir, f"{model_type}{dataset}.pth")
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # ===========================
    #       REGISTER HOOKS
    # ===========================
    latent_buffers = {}

    def make_hook(key):
        def _hook(module, input, output):
            latent_buffers[key] = output
        return _hook

    hook_handles = []
    for name in latent_module_names:
        if not hasattr(model, name):
            raise AttributeError(f"Model has no attribute '{name}' for latent hook.")
        module = getattr(model, name)
        handle = module.register_forward_hook(make_hook(name))
        hook_handles.append(handle)

    # ===========================
    #       METRIC OBJECTS
    # ===========================
    dice_metric = DiceScore(num_classes=1).to(device)

    all_dice = []
    all_ravd = []
    global_idx = 0

    # ===========================
    #       EVALUATION LOOP
    # ===========================
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing"):
            images = batch_data["image"].to(device)   # [B, C, H, W]
            labels = batch_data["label"].to(device)   # [B, 1, H, W]
            B, _, H, W = images.shape

            latent_buffers.clear()

            _, _, logits = model(
                x=images,
                class_definitions=class_definitions,
                device=device,
            )  # [B, 1, H, W]

            # ----- SEGMENTATION METRICS -----
            dice_val = dice_metric(logits, labels).item()
            all_dice.append(dice_val)

            seg_probs = torch.sigmoid(logits)
            seg_bin   = (seg_probs >= 0.5).long().cpu().numpy()
            gts       = labels.cpu().numpy().astype(np.int64)

            seg_bin_2d = seg_bin[:, 0, :, :]
            gts_2d     = gts[:, 0, :, :]

            batch_ravd = ravd_batch(gts_2d, seg_bin_2d)
            all_ravd.extend(batch_ravd.tolist())

            # ==============================
            #   BUILD LATENT MAPS (H x W)
            # ==============================
            latent_maps = {}

            for key, feat in latent_buffers.items():
                if feat.dim() != 4:
                    raise RuntimeError(f"Latent '{key}' has shape {feat.shape}, expected 4D [B, C, h, w].")
                up = F.interpolate(
                    feat,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                lat = up.mean(dim=1, keepdim=True)

                lat_min = lat.amin(dim=(2, 3), keepdim=True)
                lat_max = lat.amax(dim=(2, 3), keepdim=True)
                lat_norm = (lat - lat_min) / (lat_max - lat_min + 1e-8)

                latent_maps[key] = lat_norm

            latent_maps["softmask"] = seg_probs

            # ==============================
            #   GET FILENAMES FOR SAVING
            # ==============================
            # PDAC2DDataset now returns "image_path"
            raw_paths = batch_data["image_path"]  # from your dataset

            # DataLoader will collate this into a list when batch_size>1
            if isinstance(raw_paths, (list, tuple)):
                paths = list(raw_paths)
            else:
                paths = [raw_paths]

            # (Optional one-time debug)
            if global_idx == 0:
                print("batch_data keys:", list(batch_data.keys()))
                print("image_path batch:", paths)

            # ==============================
            #       SAVE LATENTS AS TIF
            # ==============================
            for b in range(B):
                if paths[b] is not None:
                    base_name = os.path.splitext(os.path.basename(str(paths[b])))[0]
                else:
                    base_name = f"sample_{global_idx}"
                global_idx += 1

                for key, tens in latent_maps.items():
                    latent_np = tens[b, 0].cpu().numpy().astype(np.float32)
                    out_dir = latent_dirs[key]
                    out_path = os.path.join(out_dir, base_name + ".tif")
                    tiff.imwrite(out_path, latent_np)

    for h in hook_handles:
        h.remove()

    avg_dice = float(np.mean(all_dice)) if len(all_dice) > 0 else 0.0
    avg_ravd = float(np.mean(all_ravd)) if len(all_ravd) > 0 else 0.0

    print("\n=== Test metrics on PDAC 2D ===")
    print(f"Average DSC  (segmentation mask) : {avg_dice * 100:.2f}%")
    print(f"Average RAVD                     : {avg_ravd:.4f}")
    print(f"Latent TIFs saved under          : {latents_root}")
    for name, d in latent_dirs.items():
        print(f"  - {name}: {d}")



if __name__ == "__main__":
    main()
