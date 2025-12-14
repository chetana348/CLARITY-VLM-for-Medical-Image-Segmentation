import os
import glob
import random
import pandas as pd
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor


# -------------------------------------------------
# 1) Load per-image descriptions from CSV
# -------------------------------------------------
def load_description_map(desc_csv_path):
    """
    Expects a CSV with at least:
        image_id, description

    image_id is usually the filename stem, e.g. 'img_0001' for 'img_0001.tif'
    Returns: dict[str, str] mapping image_id -> description
    """
    df = pd.read_csv(desc_csv_path)
    desc_map = {}

    for _, row in df.iterrows():
        key = str(row["image_id"])
        desc = str(row["description"])
        desc_map[key] = desc

    return desc_map


# -------------------------------------------------
# 2) PDAC slice dataset with optional descriptions
# -------------------------------------------------
class Dataset(Dataset):
    def __init__(
        self,
        image_paths,
        mask_paths,
        desc_map,
        augment: bool = False,
        
        use_basename: bool = True,
    ):
        """
        image_paths: list of image file paths
        mask_paths : list of mask file paths (same length)
        augment    : simple flips/rotations
        desc_map   : dict mapping image_id -> description (optional)
        use_basename:
            if True, use basename without extension as key, e.g.
            '.../img_0001.tif' -> 'img_0001'
        """
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.to_tensor = ToTensor()

        self.desc_map = desc_map if desc_map is not None else {}
        self.use_basename = use_basename

    def __len__(self):
        return len(self.image_paths)

    def _get_image_id(self, img_path: str) -> str:
        """
        Convert full path -> key used in desc_map.
        Default: basename without extension.
        """
        if self.use_basename:
            fname = os.path.basename(img_path)
            stem, _ = os.path.splitext(fname)
            return stem
        else:
            # you can choose to key by full path instead if you want
            return img_path

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # ----- load images -----
        image = Image.open(img_path).convert("F")   # float grayscale
        mask  = Image.open(mask_path).convert("L")  # label in 0/1

        # ----- simple augmentations -----
        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            if random.random() < 0.5:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)

            # random 90° rotation
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                angle = 90 * k
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask  = TF.rotate(mask,  angle, interpolation=TF.InterpolationMode.NEAREST)

        # ----- to tensors -----
        image = self.to_tensor(image)  # [1, H, W]
        mask  = self.to_tensor(mask)   # [1, H, W]

        # binarize mask
        mask = (mask > 0).float()

        # ----- per-image description (if available) -----
        description = ""
        if self.desc_map:
            image_id = self._get_image_id(img_path)
            description = self.desc_map.get(image_id, "")

        return {
            "image": image,          # [1, H, W]
            "label": mask,           # [1, H, W]
            "image_path": img_path,  # for saving / analysis
            "description": description,  # 2–3 sentence string (possibly "")
        }


# -------------------------------------------------
# 3) DataGen with descriptions + class CSV
# -------------------------------------------------
def DataGen(
    seed,
    root_dir="/users/PAS3110/sephora20/workspace/PDAC/data/pdac_osu/cropped/ktrans",
    desc_csv=None,
    augment_train: bool = False,
):
    """
    Returns:
        train_ds, val_ds, test_ds, class_definitions_path

    - class_definitions_path still points to your original
      classes_definitions_pdac.csv (class-level PDAC vs BG metadata).
    - desc_csv (if given) is used to attach per-image descriptions.
    """
    random.seed(seed)

    # -------- TRAIN paths --------
    train_img_dir = os.path.join(root_dir, "train", "images")
    train_lbl_dir = os.path.join(root_dir, "train", "labels")

    train_images = sorted(glob.glob(os.path.join(train_img_dir, "*.tif")))
    train_labels = [os.path.join(train_lbl_dir, os.path.basename(p)) for p in train_images]

    # -------- VAL paths (use test set) --------
    val_img_dir = os.path.join(root_dir, "test", "images")
    val_lbl_dir = os.path.join(root_dir, "test", "labels")

    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.tif")))
    val_labels = [os.path.join(val_lbl_dir, os.path.basename(p)) for p in val_images]

    # same val set also returned as test
    test_images = val_images
    test_labels = val_labels

    # -------- Description map (optional) --------
    desc_map = None
    if desc_csv is not None:
        desc_map = load_description_map(desc_csv)

    # -------- Dataset objects --------
    train_ds = Dataset(
        train_images,
        train_labels,
        augment=augment_train,
        desc_map=desc_map,
        use_basename=True,  # expects image_id = basename w/o extension
    )

    val_ds = Dataset(
        val_images,
        val_labels,
        augment=False,
        desc_map=desc_map,
        use_basename=True,
    )

    test_ds = Dataset(
        test_images,
        test_labels,
        augment=False,
        desc_map=desc_map,
        use_basename=True,
    )

    # Original class-level CSV (unchanged)
    class_definitions = "/users/PAS3110/sephora20/workspace/PDAC/LLMs/stage1/texts/classes_definitions_pdac.csv"

    return train_ds, val_ds, test_ds, class_definitions

