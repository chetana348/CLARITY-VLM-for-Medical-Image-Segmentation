import torch
from monai.transforms import MapTransform
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn

def l1_loss(pred_map, target_map):
    """
    pred_map:   [B, 1, H1, W1]
    target_map: [B, 1, H2, W2]

    If spatial sizes differ, we resize target_map to match pred_map
    and then compute L1 loss.
    """
    if pred_map.ndim != 4 or target_map.ndim != 4:
        raise ValueError(
            f"l1_loss expects 4D tensors [B, C, H, W], "
            f"got {pred_map.shape} and {target_map.shape}"
        )

    B1, C1, H1, W1 = pred_map.shape
    B2, C2, H2, W2 = target_map.shape

    if B1 != B2 or C1 != C2:
        raise ValueError(
            f"Batch/channel mismatch in l1_loss: "
            f"{pred_map.shape} vs {target_map.shape}"
        )

    if (H1, W1) != (H2, W2):
        target_map = F.interpolate(
            target_map, size=(H1, W1), mode="bilinear", align_corners=False
        )

    return F.l1_loss(pred_map, target_map, reduction="mean")



def info_nce_loss(
    visual_embeddings,      # e.g. [B, C, H, W] or [B, C, L]
    text_embeddings,        # e.g. [B, L_t, D] or [B, D]
    cp_feats_visual,        # nn.Linear(in_features=num_channels, out_features=128)
    cp_feats_text,          # nn.Linear(in_features=projection_dim, out_features=128)
    device,
    temperature: float = 0.2,
):
    """
    Symmetric InfoNCE (image<->text) with robust shape handling.
    """

    # ---- 1. Visual: pool to [B, C] ----
    if visual_embeddings.dim() == 4:
        # [B, C, H, W] -> [B, C]
        v_vec = visual_embeddings.mean(dim=(2, 3))
    elif visual_embeddings.dim() == 3:
        # [B, C, L] -> [B, C]
        v_vec = visual_embeddings.mean(dim=2)
    else:
        raise ValueError(f"Unexpected visual_embeddings shape: {visual_embeddings.shape}")

    # ---- 2. Text: pool to [B, D] ----
    if isinstance(text_embeddings, (list, tuple)):
        # If it's a list of [B, L, D] / [B, D] tensors, average them
        t_vecs = []
        for te in text_embeddings:
            if te.dim() == 3:
                # [B, L, D] -> [B, D]
                t_vecs.append(te.mean(dim=1))
            elif te.dim() == 2:
                # [B, D]
                t_vecs.append(te)
            else:
                raise ValueError(f"Unexpected text_embeddings element shape: {te.shape}")
        # Average across list dimension -> [B, D]
        t_vec = torch.stack(t_vecs, dim=0).mean(dim=0)
    else:
        # Single tensor
        if text_embeddings.dim() == 3:
            # [B, L, D] -> [B, D]
            t_vec = text_embeddings.mean(dim=1)
        elif text_embeddings.dim() == 2:
            # [B, D]
            t_vec = text_embeddings
        else:
            raise ValueError(f"Unexpected text_embeddings shape: {text_embeddings.shape}")

    # ---- 3. Ensure same batch size B for both streams ----
    B_v = v_vec.size(0)
    B_t = t_vec.size(0)
    B = min(B_v, B_t)
    v_vec = v_vec[:B]
    t_vec = t_vec[:B]

    # ---- 4. Linear projections into shared space [B, 128] ----
    v_proj = cp_feats_visual(v_vec.to(device))   # in_features must match v_vec.size(1)
    t_proj = cp_feats_text(t_vec.to(device))     # in_features must match t_vec.size(1)

    # Normalize
    v_proj = F.normalize(v_proj, dim=-1)  # [B, 128]
    t_proj = F.normalize(t_proj, dim=-1)  # [B, 128]

    # ---- 5. Similarity logits [B, B] ----
    logits = (v_proj @ t_proj.t()) / temperature  # [B, B]

    # ---- 6. Labels that match logits *in each direction* ----
    labels_i2t = torch.arange(logits.size(0), device=device)  # [B]
    labels_t2i = torch.arange(logits.size(1), device=device)  # [B] (same here, but robust)

    # ---- 7. Symmetric InfoNCE (image->text, text->image) ----
    loss_i2t = F.cross_entropy(logits, labels_i2t)       # logits: [B, B], labels: [B]
    loss_t2i = F.cross_entropy(logits.t(), labels_t2i)   # logits.T: [B, B], labels: [B]

    return 0.5 * (loss_i2t + loss_t2i)
    
def support_loss(
    visual_embeddings,      # e.g. [B, C, H, W] or [B, C, L]
    text_embeddings,        # list of [B, L, D] / [B, D] OR a single tensor
    cp_feats_visual,        # nn.Linear(in_features=num_channels, out_features=128)
    cp_feats_text,          # nn.Linear(in_features=projection_dim, out_features=128)
    device,
    temperature: float = 0.2,   # kept in signature, not strictly used here
):
    """
    Support alignment loss:
    - Pools visual and text to per-image vectors.
    - Projects into same latent dim.
    - Encourages them to be similar via cosine similarity.

    Returns a scalar loss.
    """

    # ---- 1. Visual: pool to [B, C] ----
    if visual_embeddings.dim() == 4:
        # [B, C, H, W] -> [B, C]
        v_vec = visual_embeddings.mean(dim=(2, 3))
    elif visual_embeddings.dim() == 3:
        # [B, C, L] -> [B, C]
        v_vec = visual_embeddings.mean(dim=2)
    else:
        raise ValueError(f"Unexpected visual_embeddings shape: {visual_embeddings.shape}")

    # ---- 2. Text: pool to [B, D] ----
    if isinstance(text_embeddings, (list, tuple)):
        # List of tensors: average over list
        t_vecs = []
        for te in text_embeddings:
            if te.dim() == 3:
                # [B, L, D] -> [B, D]
                t_vecs.append(te.mean(dim=1))
            elif te.dim() == 2:
                # [B, D]
                t_vecs.append(te)
            else:
                raise ValueError(f"Unexpected text_embeddings element shape: {te.shape}")
        # [num_lists, B, D] -> [B, D]
        t_vec = torch.stack(t_vecs, dim=0).mean(dim=0)
    else:
        # Single tensor
        te = text_embeddings
        if te.dim() == 3:
            # [B, L, D] -> [B, D]
            t_vec = te.mean(dim=1)
        elif te.dim() == 2:
            # [B, D]
            t_vec = te
        else:
            raise ValueError(f"Unexpected text_embeddings shape: {te.shape}")

    # ---- 3. Ensure same batch size ----
    B_v = v_vec.size(0)
    B_t = t_vec.size(0)
    B = min(B_v, B_t)
    v_vec = v_vec[:B]
    t_vec = t_vec[:B]

    # ---- 4. Project into shared latent space ----
    v_proj = cp_feats_visual(v_vec.to(device))   # e.g. [B, 128]
    t_proj = cp_feats_text(t_vec.to(device))     # e.g. [B, 128]

    # Normalize
    v_proj = F.normalize(v_proj, dim=-1)
    t_proj = F.normalize(t_proj, dim=-1)

    # ---- 5. Cosine similarity alignment loss ----
    cos_sim = F.cosine_similarity(v_proj, t_proj, dim=-1)  # [B]
    # Maximize similarity -> minimize (1 - cos_sim)
    loss = 1.0 - cos_sim.mean()

    return loss