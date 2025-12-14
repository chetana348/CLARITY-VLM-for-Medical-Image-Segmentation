#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import tifffile as tiff

from dataset import DataGen
from models.clinvlm import ClinicalVLM


# ===========================
#         HELPERS
# ===========================

def seed_all(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def dice_binary_mask(pred_bin: np.ndarray, gt_bin: np.ndarray, smooth: float = 1.0) -> float:
    """
    pred_bin, gt_bin: {0,1} arrays, shape [H,W]
    """
    pred_bin = pred_bin.astype(np.float32)
    gt_bin   = gt_bin.astype(np.float32)
    inter = (pred_bin * gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    return float((2.0 * inter + smooth) / (denom + smooth))

def ravd_binary_mask(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-6) -> float:
    """
    RAVD = |V_pred - V_gt| / (V_gt + eps)
    If GT empty: 0 if pred empty else 1
    """
    gt_area   = float((gt_bin == 1).sum())
    pred_area = float((pred_bin == 1).sum())
    if gt_area == 0.0:
        return 0.0 if pred_area == 0.0 else 1.0
    return float(abs(pred_area - gt_area) / (gt_area + eps))

def get_paths_from_batch(batch_data, B):
    """
    Robust filename extraction.
    Tries MONAI meta and common dataset keys.
    """
    paths = [None] * B

    # ---- 1) MONAI-style meta dict ----
    if "image_meta_dict" in batch_data:
        meta = batch_data["image_meta_dict"]

        # dict of lists
        if isinstance(meta, dict) and "filename_or_obj" in meta:
            raw = meta["filename_or_obj"]
            if isinstance(raw, (list, tuple)):
                for i in range(min(B, len(raw))):
                    paths[i] = raw[i]
            else:
                paths = [raw] * B

        # list of dicts
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

    # ---- 2) Common custom dataset keys ----
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

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


# ===========================
#            MAIN
# ===========================

def main():
    # ---------------------------
    # Config (EDIT THESE)
    # ---------------------------
    seed = 42

    # Use the same description CSV + checkpoint as the experiment you trained
    desc_csv  = "/users/PAS3110/sephora20/workspace/PDAC/LLMs/stage1/texts/pdac_ktrans_descriptions.csv"

    # Your trained checkpoint (must correspond to out_channels=2 run)
    ckpt_path = "./outputs/pdac_ktrans/weights/model.pth"

    # Where to save outputs
    out_root  = "./outputs/test"
    ensure_dir(out_root)

    # Dataloader
    batch_size  = 1
    num_workers = 4

    # Latent hooks: ONLY include modules that реально exist as attributes on ClinicalVLM
    latent_module_names = [ #"encoder1", 
            #"encoder4", 
            #"encoder10",
            "decoder2" ,
            #"decoder1", 
            ]

    # Classes: background=0, PZ=1, TZ=2 (your training dict)
    class_ids = [1]
    class_names = {1: "tumor"}

    # ---------------------------
    # Setup
    # ---------------------------
    seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---------------------------
    # Load dataset
    # ---------------------------
    train, val, test, class_definitions = DataGen(seed, desc_csv=desc_csv)

    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Infer in_channels like training
    sample = test[0]
    in_channels = sample["image"].shape[0]

    num_channels = 2  

    model = ClinicalVLM(
        img_size=(128, 128),
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=1,
        feature_size=48,
        device=device,
        text_encoder="pubmedbert",
        use_checkpoint=True,
        use_v2=True,
    ).to(device)

    # ---------------------------
    # Load checkpoint
    # ---------------------------
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Your training saved dict: {'model': state_dict, 'cp_feats_*':..., 'log_sigma*':...}
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    # ---------------------------
    # Register hooks (optional)
    # ---------------------------
    latent_buffers = {}

    def make_hook(key):
        def _hook(module, inp, out):
            latent_buffers[key] = out
        return _hook

    hook_handles = []
    for name in latent_module_names:
        if not hasattr(model, name):
            raise AttributeError(f"Model has no attribute '{name}' for latent hook. "
                                 f"Either remove it or use the correct module name.")
        handle = getattr(model, name).register_forward_hook(make_hook(name))
        hook_handles.append(handle)

    # ---------------------------
    # Output dirs
    # ---------------------------
    #prob_dir   = ensure_dir(os.path.join(out_root, "probs"))
    #pred_dir   = ensure_dir(os.path.join(out_root, "pred_masks"))
    latent_dir = ensure_dir(os.path.join(out_root, "latents"))

    # per-latent subdirs
    latent_subdirs = {k: ensure_dir(os.path.join(latent_dir, k)) for k in latent_module_names}

    # ---------------------------
    # Metrics accumulators
    # ---------------------------
    dice_per_class = {c: [] for c in class_ids}
    ravd_per_class = {c: [] for c in class_ids}
    dice_mean_all  = []
    ravd_mean_all  = []

    global_idx = 0

    # ---------------------------
    # Evaluation loop
    # ---------------------------
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing"):
            images = batch_data["image"].to(device)   # [B,C,H,W]
            labels = batch_data["label"].to(device)   # often [B,1,H,W] with {0,1,2}
            B, _, H, W = images.shape

            # normalize labels shape -> [B,H,W] int
            if labels.ndim == 4 and labels.shape[1] == 1:
                labels_int = labels[:, 0].long()
            else:
                labels_int = labels.long()

            latent_buffers.clear()

            # IMPORTANT: unpack 4 outputs
            feats, class_text_feats, logits, img_text_emb = model(
                x=images,
                class_definitions=class_definitions,
                image_descriptions=None,
                device=device,
            )  # logits: [B,2,H,W]

            # softmax probs
            probs = torch.softmax(logits, dim=1)      # [B,2,H,W]
            pred  = torch.argmax(probs, dim=1)        # [B,H,W] values 0/1

            bg_thresh = 0.5
            max_prob, arg = torch.max(probs, dim=1)   # max over 2 channels
            pred_012 = torch.where(max_prob < bg_thresh, torch.zeros_like(arg), arg + 1)  # 0,1,2

            # convert to numpy for metrics + saving
            probs_np = probs.cpu().numpy()            # [B,2,H,W]
            pred_np  = pred_012.cpu().numpy().astype(np.uint8)   # [B,H,W]
            gt_np    = labels_int.cpu().numpy().astype(np.uint8) # [B,H,W]

            # filenames
            paths = get_paths_from_batch(batch_data, B)

            for b in range(B):
                if paths[b] is not None:
                    base = os.path.splitext(os.path.basename(str(paths[b])))[0]
                else:
                    base = f"sample_{global_idx}"
                global_idx += 1

                # ---- Save prob maps (PZ/TZ) ----
                # channel0 -> PZ prob, channel1 -> TZ prob
                #tiff.imwrite(os.path.join(prob_dir, f"{base}_prob_PZ.tif"),
                 #            probs_np[b, 0].astype(np.float32))
                #tiff.imwrite(os.path.join(prob_dir, f"{base}_prob_TZ.tif"),
                #             probs_np[b, 1].astype(np.float32))

                # ---- Save predicted mask (0/1/2) ----
                #tiff.imwrite(os.path.join(pred_dir, f"{base}_pred_mask_012.tif"),
                #             pred_np[b].astype(np.uint8))

                # ---- Save hooked latents (optional) ----
                for key, feat in latent_buffers.items():
                    if not isinstance(feat, torch.Tensor) or feat.ndim != 4:
                        continue  # skip unexpected outputs

                    up = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
                    lat = up.mean(dim=1, keepdim=False)  # [B,H,W]

                    # per-sample minmax
                    lat_b = lat[b]
                    lat_min = float(lat_b.min().item())
                    lat_max = float(lat_b.max().item())
                    lat_norm = (lat_b - lat_min) / (lat_max - lat_min + 1e-8)

                    tiff.imwrite(os.path.join(latent_subdirs[key], f"{base}.tif"),
                                 lat_norm.detach().cpu().numpy().astype(np.float32))

            # ---- Metrics per sample (per class) ----
            for b in range(B):
                dice_vals = []
                ravd_vals = []
                for c in class_ids:
                    pred_c = (pred_np[b] == c).astype(np.uint8)
                    gt_c   = (gt_np[b]   == c).astype(np.uint8)

                    d = dice_binary_mask(pred_c, gt_c)
                    r = ravd_binary_mask(pred_c, gt_c)

                    dice_per_class[c].append(d)
                    ravd_per_class[c].append(r)

                    dice_vals.append(d)
                    ravd_vals.append(r)

                dice_mean_all.append(float(np.mean(dice_vals)))
                ravd_mean_all.append(float(np.mean(ravd_vals)))

    # remove hooks
    for h in hook_handles:
        h.remove()

    # ---------------------------
    # Print summary
    # ---------------------------
    print("\n============================")
    print("        TEST SUMMARY")
    print("============================")

    for c in class_ids:
        d_mean = float(np.mean(dice_per_class[c])) if len(dice_per_class[c]) else 0.0
        r_mean = float(np.mean(ravd_per_class[c])) if len(ravd_per_class[c]) else 0.0
        print(f"{class_names[c]} (class={c}): Dice={d_mean*100:.2f}%, RAVD={r_mean:.4f}")

    d_all = float(np.mean(dice_mean_all)) if len(dice_mean_all) else 0.0
    r_all = float(np.mean(ravd_mean_all)) if len(ravd_mean_all) else 0.0
    print(f"\nMEAN over classes: Dice={d_all*100:.2f}%, RAVD={r_all:.4f}")

    print("\nSaved outputs:")
    print(f"  latents    -> {latent_dir}")



if __name__ == "__main__":
    main()
