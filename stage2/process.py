import os
import numpy as np
import tifffile as tiff
from PIL import Image
from glob import glob

# -----------------------------
# CONFIG
# -----------------------------
splits = ["train", "test", "val"]  # you can add "val" etc. if needed

# Map "tag" -> subfolder name in your latents directory.
# Adjust the values on the right if your folder names differ.
latent_variants = {
    #"encoder1":  "encoder1",
    #"encoder4":  "encoder4",
    #"encoder10": "encoder10",
    #"decoder1":  "decoder1",
    "decoder2":  "decoder2",
    #"softmask":  "softmask",   # change this if your softmask dir is named differently
}

# Root paths (adjust if needed)
images_root   = "/users/PAS3110/sephora20/workspace/PDAC/data/panther/cropped"
latents_root  = "/users/PAS3110/sephora20/workspace/PDAC/LLMs/stage1/outputs/pdac_ktransnn_on_panther/ten/latents"

# If softmasks live somewhere else, set this and handle below;
# otherwise we’ll just reuse latents_root.
softmask_root = None  # e.g. "/users/.../stage2/outputs/pdac_mri/softmasks"


# -----------------------------
# Helper functions
# -----------------------------
def norm_image_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.max() > 1.0:  # typical uint8 0..255
        x = x / 255.0
    else:
        x = np.clip(x, 0.0, 1.0)
    return x

def compute_latent_scaling(latent_dir: str, max_samples: int = 200):
    latent_files = sorted(glob(os.path.join(latent_dir, "*.tif")))
    if len(latent_files) == 0:
        raise FileNotFoundError(f"No .tif latent files found in {latent_dir}")

    samples = []
    for i, lp in enumerate(latent_files):
        # Subsample to at most ~max_samples files for speed
        if len(latent_files) > max_samples:
            step = max(1, len(latent_files) // max_samples)
            if i % step != 0:
                continue

        with Image.open(lp) as L:
            arr = np.array(L)
        if arr.ndim == 3:  # squeeze if (H,W,1)
            arr = arr[..., 0]
        arr = arr.astype(np.float32)
        samples.append(arr.ravel())

    if not samples:
        raise RuntimeError(f"No samples collected from {latent_dir} to estimate percentiles")

    samples = np.concatenate(samples)
    p1, p99 = np.percentile(samples, [1, 99])

    # safety
    if p99 <= p1:
        p1, p99 = float(samples.min()), float(samples.max())
        if p99 == p1:
            p1, p99 = 0.0, 1.0

    print(f"[latent scaling] {latent_dir}")
    print(f"  p1={p1:.6g}, p99={p99:.6g}, Nfiles={len(latent_files)}")
    return p1, p99

def norm_latent(x: np.ndarray, p1: float, p99: float) -> np.ndarray:
    x = x.astype(np.float32)
    x = (x - p1) / (p99 - p1 + 1e-8)
    x = np.clip(x, 0.0, 1.0)
    return x


# -----------------------------
# Main processing per (split, variant)
# -----------------------------
def process_split_and_variant(split: str, tag: str, latent_subdir: str):
    # Paths
    images_dir = os.path.join(images_root, split, "images")

    # Decide latent dir (softmask can be separate)
    if tag == "softmask" and softmask_root is not None:
        latents_dir = os.path.join(softmask_root, split, latent_subdir)
    else:
        latents_dir = os.path.join(latents_root, split, latent_subdir)

    out_dir = os.path.join(
        images_root, split, "CLARiTY", "ten", f"images_pdac_ktransnn_on_panther_{tag}"
    )
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(f"Split: {split} | Variant: {tag}")
    print(f"  images_dir : {images_dir}")
    print(f"  latents_dir: {latents_dir}")
    print(f"  out_dir    : {out_dir}")

    # 1) Robust dataset-level scaling for this latent_dir
    try:
        p1, p99 = compute_latent_scaling(latents_dir)
    except FileNotFoundError as e:
        print(f"!! Skipping {split}/{tag}: {e}")
        return
    except RuntimeError as e:
        print(f"!! Skipping {split}/{tag}: {e}")
        return

    # 2) Convert & save all pairs
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".tif")]
    if not image_files:
        print(f"!! No .tif images found in {images_dir}, skipping.")
        return

    for img_name in sorted(image_files):
        base = os.path.splitext(img_name)[0]
        latent_name = f"{base}.tif"

        img_path = os.path.join(images_dir, img_name)
        latent_path = os.path.join(latents_dir, latent_name)

        if not os.path.exists(latent_path):
            print(f"⚠️  No matching latent for {img_name}, skipping.")
            continue

        # Load image
        with Image.open(img_path) as I:
            img = np.array(I)  # (H,W) uint8/uint16 or (H,W,1/3)

        # Load latent
        with Image.open(latent_path) as L:
            lat = np.array(L)  # (H,W) float/uint or (H,W,1)

        # Ensure shapes match: resize latent if needed
        if (lat.shape[1], lat.shape[0]) != (img.shape[1], img.shape[0]):
            L = Image.fromarray(lat)
            L = L.resize((img.shape[1], img.shape[0]), resample=Image.BILINEAR)
            lat = np.array(L)

        # Squeeze channels if needed
        if img.ndim == 3:
            if img.shape[2] == 1:
                img = img[..., 0]
            else:
                img = img.mean(axis=2)  # average if some RGB slipped in
        if lat.ndim == 3 and lat.shape[2] == 1:
            lat = lat[..., 0]

        # Normalize per channel
        img_n = norm_image_uint8(img)       # -> float32 [0,1]
        lat_n = norm_latent(lat, p1, p99)   # -> float32 [0,1]

        # Stack (C,H,W) float32
        combined = np.stack([img_n, lat_n], axis=0).astype(np.float32)

        # Save as compressed BigTIFF with channel axis metadata
        out_path = os.path.join(out_dir, base + ".tif")
        tiff.imwrite(
            out_path,
            combined,
            dtype=np.float32,
            bigtiff=True,
            compression="zlib",
            photometric="minisblack",
            metadata={"axes": "CYX"},  # channel, y, x
        )
        print(f"Saved: {out_path} | shape={combined.shape}, dtype={combined.dtype}")


# -----------------------------
# Run for all splits & variants
# -----------------------------
if __name__ == "__main__":
    for split in splits:
        for tag, latent_subdir in latent_variants.items():
            process_split_and_variant(split, tag, latent_subdir)
