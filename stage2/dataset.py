import os, json, random
import numpy as np
import tifffile as tiff
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path

def load_prompts(path):
    """Load prompts from .json (preferred) or .txt formatted as 'stem: text'."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)  # {stem: prompt}
    # txt fallback
    prompts = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            stem, text = line.split(":", 1)
            prompts[stem.strip()] = text.strip()
    return prompts

class Data_Gen(Dataset):
    def __init__(
        self,
        data_path,                  # folder with *_combined.tif (C,H,W) float32, C=2
        label_path,                 # folder with masks (H,W)
        prompt_file=None,           # path to prompts.json or prompts.txt
        size=224,                   # resize target (int or (h,w))
        transform=None,             # optional external transform; expects CHW torch or HWC numpy (your choice)
        mode='train',
        strict_pairing=True         # if True, pair by stem; else rely on sorted order
    ):
        self.data_path = Path(data_path)
        self.label_path = Path(label_path)
        self.transform = transform
        self.mode = mode
        self.size = (size, size) if isinstance(size, int) else tuple(size)

        # Index by stem to be robust
        img_files = [f for f in os.listdir(self.data_path) if f.lower().endswith(".tif")]
        lbl_files = [f for f in os.listdir(self.label_path) if f.lower().endswith((".tif", ".png"))]

        def stem(f): return os.path.splitext(f)[0].replace("", "")
        imgs_by_stem = {stem(f): f for f in img_files}
        lbls_by_stem = {os.path.splitext(f)[0]: f for f in lbl_files}

        # Intersect keys (paired samples)
        common = sorted(set(imgs_by_stem.keys()) & set(lbls_by_stem.keys())) if strict_pairing else \
                 sorted([stem(f) for f in img_files])

        self.items = [(imgs_by_stem[s], lbls_by_stem.get(s, None), s) for s in common]
        if strict_pairing:
            self.items = [(i, l, s) for (i, l, s) in self.items if l is not None]

        # Load prompts dict if provided
        self.prompts = load_prompts(prompt_file) if prompt_file else None

        print(f"Paired samples: {len(self.items)} (images in dir={len(img_files)}, labels in dir={len(lbl_files)})")
        if prompt_file:
            print(f"Prompts loaded: {len(self.prompts)} from {prompt_file}")

    def __len__(self):
        return len(self.items)

    def _read_image_chw(self, path):
        """
        Read TIFF that may be saved as (C,H,W) or (H,W,C) or (H,W).
        Returns float32 CHW.
        """
        arr = tiff.imread(str(path))
        arr = np.asarray(arr)

        # If (H,W) -> add channel
        if arr.ndim == 2:
            arr = arr[None, ...]  # (1,H,W)
        elif arr.ndim == 3:
            # Heuristic: if first dim is small (<=4), assume CHW; otherwise HWC
            if arr.shape[0] <= 4:
                pass  # already CHW
            else:
                arr = np.moveaxis(arr, -1, 0)  # HWC->CHW
        else:
            raise ValueError(f"Unexpected image shape {arr.shape} for {path}")

        # Ensure float32
        arr = arr.astype(np.float32, copy=False)
        return arr  # (C,H,W)

    def _read_mask(self, path):
        m = tiff.imread(str(path))
        m = np.asarray(m)
        if m.ndim == 3 and m.shape[-1] == 1:
            m = m[..., 0]
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        # binarize robustly (assume >0 means foreground)
        m = (m > 0).astype(np.uint8)
        return m  # (H,W) uint8 {0,1}

    def __getitem__(self, idx):
        img_name, lbl_name, stem = self.items[idx]
        img_path = self.data_path / img_name
        label_path = self.label_path / lbl_name

        # --- Load ---
        image = self._read_image_chw(img_path)      # (C,H,W), float32
        label = self._read_mask(label_path)         # (H,W),   uint8

        # --- Resize ---
        # cv2 expects HWC; do per-channel resizing safely
        C, H, W = image.shape
        image_hwc = np.moveaxis(image, 0, -1)       # (H,W,C)
        image_hwc = cv2.resize(image_hwc, self.size, interpolation=cv2.INTER_LINEAR)
        image = np.moveaxis(image_hwc, -1, 0)       # back to (C,h,w)

        label = cv2.resize(label.astype(np.uint8), self.size, interpolation=cv2.INTER_NEAREST)
        
        label = (label > 0).astype(np.uint8)

        # --- Augmentations (flip/rotate) ---
        if self.mode == 'train':
            if random.random() > 0.5:
                image = np.flip(image, axis=2).copy()   # horiz
                label = np.flip(label, axis=1).copy()
            if random.random() > 0.5:
                image = np.flip(image, axis=1).copy()   # vert
                label = np.flip(label, axis=0).copy()
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                image = np.rot90(image, k=k, axes=(1, 2)).copy()
                label = np.rot90(label, k=k).copy()

        # --- Optional external transform ---
        # If your transform expects CHW torch.Tensor, convert temporarily
        if self.transform is not None:
            # You can adapt this block to your transform’s expected format.
            image_t = torch.from_numpy(image.copy()).float()  # CHW
            out = self.transform(image=image_t, mask=torch.from_numpy(label))
            image = out["image"].numpy()
            label = out["mask"].numpy()
        
        # --- To torch (CPU; move to GPU in train loop) ---
        image = torch.from_numpy(image).float()           # (C,h,w)
        label = torch.from_numpy(label.astype(np.int64))  # (h,w), long
        #print(torch.unique(label))
        #label[label==128]==2
        #label[label==255]==1
        #label[label<0]==0
        # --- Prompt string (or empty if not provided) ---
        text = ""
        if self.prompts is not None:
            text = self.prompts.get(stem, "")

        # Return: image, label, text, stem
        return image, label, text, stem, img_name
        
class Data_Gen_2p5D(Dataset):
    """
    2.5D wrapper for your original Data_Gen.

    For index i, returns a 6-channel volume:

        (prev_image, prev_latent,
         curr_image, curr_latent,
         next_image, next_latent)

    Along with: center label, center text prompt, center stem, center filename.
    """

    def __init__(self, base_dataset):
        super().__init__()
        self.base = base_dataset    # <-- original Data_Gen

    def __len__(self):
        return len(self.base)

    def _clamp(self, idx):
        """Clamp idx to dataset boundaries."""
        return max(0, min(len(self.base) - 1, idx))

    def __getitem__(self, idx):
        # ----------------------------
        # neighbors: i-1, i, i+1
        # ----------------------------
        idx_p = self._clamp(idx - 1)
        idx_c = self._clamp(idx)
        idx_n = self._clamp(idx + 1)

        # -----------------------------------------
        # original Data_Gen returns:
        #   image (2,H,W), label (H,W),
        #   text, stem, img_name
        # -----------------------------------------
        img_p, lab_p, text_p, stem_p, name_p = self.base[idx_p]
        img_c, lab_c, text_c, stem_c, name_c = self.base[idx_c]
        img_n, lab_n, text_n, stem_n, name_n = self.base[idx_n]

        # -----------------------------------------------------------------
        # 2.5D input = concatenate all 3 slices → (6,H,W)
        # img_* is already (2,H,W): [image, latent]
        # -----------------------------------------------------------------
        images_2p5d = torch.cat([img_p, img_c, img_n], dim=0)  # (6,H,W)

        # -----------------------------------------
        # Return center-slice supervision only
        # -----------------------------------------
        label_center = lab_c.unsqueeze(0).long()   # (1,H,W) for loss
        text_center  = text_c
        stem_center  = stem_c
        name_center  = name_c

        return images_2p5d, label_center, text_center, stem_center, name_center

