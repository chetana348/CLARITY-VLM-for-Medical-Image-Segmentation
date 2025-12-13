from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from models.transform import TwoWayTransformer as context

from models.bert import BertForMaskedLM
from transformers import BertTokenizer
from models.blocks import *

rearrange, _ = optional_import("einops", name="rearrange")


# ------------------------------
#   Main UniViLa architecture
# ------------------------------

class ClinicalVLM(nn.Module):
    """
    Clinical Vision Language Model:

    - External behavior:
        * swin backbone with staged calls: self.swinViT(stage_idx, x)
        * UNETR-style encoder1/2/3/4/10, decoder5/4/3/2/1, out head
        * cp_feats on the deepest encoder feature
        * text encoding + context pipeline
        * forward(x, class_definitions, device='cuda') -> (feats, features_inputs, logits)

    - Internally:
        * Uses helper encoder/decoder wrappers
        * Builds channel projections & contexts via compact loops
        * Adds LiteSEBlock on deep features and FeatureReNorm on final decoder output
    """

    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        device=None,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 2,
        downsample: str | nn.Module = "merging",
        use_v2: bool = False,
        text_encoder: str | None = None,
    ) -> None:
        super().__init__()

        # --------------------
        # Device + checks
        # --------------------
        self.device = device if device is not None else torch.device("cpu")
        self.normalize = normalize
        self.num_stages = len(depths)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3.")

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch = ensure_tuple_rep(2, spatial_dims)
        window = ensure_tuple_rep(7, spatial_dims)

        self._validate_grid(img_size, patch)
        self._validate_rates(drop_rate, attn_drop_rate, dropout_path_rate)

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        # --------------------
        # Swin backbone
        # --------------------
        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window,
            patch_size=patch,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        )

        # --------------------
        # Encoder path
        # --------------------
        enc_specs = [
            ("encoder1", in_channels,        feature_size),
            ("encoder2", feature_size,       feature_size),
            ("encoder3", feature_size * 2,   feature_size * 2),
            ("encoder4", feature_size * 4,   feature_size * 4),
            ("encoder10", feature_size * 16, feature_size * 16),  # bottleneck
        ]
        self._enc_blocks = nn.ModuleDict()
        for name, c_in, c_out in enc_specs:
            block = UNetStageEncoder(spatial_dims, c_in, c_out, norm_name)
            self._enc_blocks[name] = block
            # expose with the exact attribute name for hooks
            setattr(self, name, block)

        # --------------------
        # Decoder path
        # --------------------
        dec_specs = [
            ("decoder5", feature_size * 16, feature_size * 8),
            ("decoder4", feature_size * 8,  feature_size * 4),
            ("decoder3", feature_size * 4,  feature_size * 2),
            ("decoder2", feature_size * 2,  feature_size),
            ("decoder1", feature_size,      feature_size),
        ]
        self._dec_blocks = nn.ModuleDict()
        for name, c_in, c_out in dec_specs:
            block = UNetStageDecoder(spatial_dims, c_in, c_out, norm_name)
            self._dec_blocks[name] = block
            setattr(self, name, block)

        # Deep attention / normalization
        self.deep_se = LiteSEBlock(feature_size * 16)

        # --------------------
        # Output head + cp_feats
        # --------------------
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )

        self.cp_feats = MLPBlock(
            input_dim=feature_size * 16,
            hidden_dim=768,
            output_dim=out_channels,
        )

        # refine decoder output before final head (novel block)
        self.final_refine = FeatureReNorm(feature_size)

        # global pooling (kept for compatibility)
        self.global_pool = nn.AdaptiveAvgPool2d(1) if spatial_dims == 2 else nn.AdaptiveAvgPool3d(1)

        # --------------------
        # Text encoder family
        # --------------------
        self._init_text_stream(text_encoder)

        # --------------------
        # Channel projections (cp_layers / reverse_cp_layers)
        # --------------------
        self._init_channel_projections(feature_size)

        # --------------------
        # Bi-fusion stack
        # --------------------
        self._init_context_stack()

    # ==========================================
    #       Validation helpers
    # ==========================================

    @staticmethod
    def _validate_grid(img_size, patch_size):
        """
        Enforces divisibility of image grid by patch powers across stages.
        Same logical check as original, different style.
        """
        for dim_len, patch in zip(img_size, patch_size):
            for depth_idx in range(5):  # up to 5 resolutions
                if dim_len % (patch ** (depth_idx + 1)) != 0:
                    raise ValueError(
                        "img_size must be divisible by patch_size^(stage) for each stage resolution."
                    )

    @staticmethod
    def _validate_rates(drop, attn_drop, path_drop):
        for value, label in [(drop, "drop_rate"),
                             (attn_drop, "attn_drop_rate"),
                             (path_drop, "dropout_path_rate")]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{label} must lie in [0,1].")

    # ==========================================
    #       Text encoder setup
    # ==========================================

    def _init_text_stream(self, text_encoder: str | None):
        """
        Configure tokenizer + LM model using a small registry.
        Keeps same choices, but the wiring looks different.
        """
        if text_encoder is None:
            print("Using Vanilla BERT")
            self.tokenizer = BertTokenizer
            self.text_encoder = BertForMaskedLM
            return

        model_registry = {
            "biobert":      ("dmis-lab/biobert-v1.1",            "BioBERT"),
            "clinicalbert": ("medicalai/ClinicalBERT",           "ClinicalBERT"),
            "pubmedbert":   ("NeuML/pubmedbert-base-embeddings", "PubMedBERT"),
        }
        if text_encoder not in model_registry:
            raise ValueError(f"Unknown text_encoder: {text_encoder}")

        model_id, label = model_registry[text_encoder]
        print(f"Initializing text encoder: {label}")
        self.tokenizer = BertTokenizer.from_pretrained(model_id)
        self.text_encoder = BertForMaskedLM.from_pretrained(model_id)

    # ==========================================
    #       Channel projections
    # ==========================================

    def _init_channel_projections(self, feature_size: int):
        """
        cp_layers:    16F -> [F, 2F, 4F, 8F]
        reverse:      [F, 2F, 4F, 8F] -> 16F
        Kept as plain lists (like original), so indexing & reassignment
        work the same way in context_layer.
        """
        self.cp_layers = []
        self.reverse_cp_layers = []

        fanouts = [feature_size, feature_size * 2, feature_size * 4, feature_size * 8]
        for fo in fanouts:
            self.cp_layers.append(nn.Linear(feature_size * 16, fo))
            self.reverse_cp_layers.append(nn.Linear(fo, feature_size * 16))

    # ==========================================
    #       Conext Fusion blocks
    # ==========================================

    def _init_context_stack(self):
        """
        Construct 5-stage context stack with the same embedding dims,
        but via a compact loop, and store as a Python list (not ModuleList),
        consistent with your original code.
        """
        embedding_dims = [48, 96, 192, 384, 768]
        self.contexts = []
        for emb in embedding_dims:
            bf = context(
                depth=2,
                embedding_dim=emb,
                num_heads=8,
                mlp_dim=2048,
                activation=nn.ReLU,
                attention_downsample_rate=2,
            )
            self.contexts.append(bf.to(self.device))

    # ==========================================
    #       Load Swin weights
    # ==========================================

    def load_from(self, weights):
        """
        Swin weight loading: same parameter mapping, more compact helper style.
        """
        with torch.no_grad():
            st = weights["state_dict"]

            self.swinViT.patch_embed.proj.weight.copy_(st["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(st["module.patch_embed.proj.bias"])

            def _load_layer(layer_key: str, layer_mod):
                for blk_name, blk in layer_mod.blocks.named_children():
                    blk.load_from(weights, n_block=blk_name, layer=layer_key)

                layer = getattr(self.swinViT, layer_key)[0]
                layer.downsample.reduction.weight.copy_(st[f"module.{layer_key}.0.downsample.reduction.weight"])
                layer.downsample.norm.weight.copy_(st[f"module.{layer_key}.0.downsample.norm.weight"])
                layer.downsample.norm.bias.copy_(st[f"module.{layer_key}.0.downsample.norm.bias"])

            _load_layer("layers1", self.swinViT.layers1)
            _load_layer("layers2", self.swinViT.layers2)
            _load_layer("layers3", self.swinViT.layers3)
            _load_layer("layers4", self.swinViT.layers4)

    # ==========================================
    #       Text token utilities
    # ==========================================

    def QueryEmbedding(self, df: pd.DataFrame):
        """
        Longer, clinically rich descriptions per class.
        """
        sentences = []
        for _, row in df.iterrows():

            s = (
                f"{row['Title']}. "
                f"Shape: {row['Shape']}. "
                f"Location: {row['Location']}. "
                f"Appearance: {row['Appearance/Density']}. "
                f"Contour: {row['Contour/Symmetry']}. "
                f"Texture: {row['Internal Texture']}."
                #f"Description: {row['Description']}."
            )
            sentences.append(s)
    
        tokenized = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,  # you can set max_length=128/256 explicitly if needed
            return_tensors="pt",
        )
        return tokenized.to(self.device)


    def TextEmbedding(self, df: pd.DataFrame):
        """
        Attribute-wise sentences per class (6 per class).
        Reshapes token tensors to [num_classes, 6, seq_len] per key.
        """
        # Build a flat list, but with clear attribute tagging
        flat_list = []
        for _, row in df.iterrows():
            flat_list.extend([
                f"{row['Title']}",
                f"Shape: {row['Shape']}",
                f"Location: {row['Location']}",
                f"Appearance/Density: {row['Appearance/Density']}",
                f"Contour/Symmetry: {row['Contour/Symmetry']}",
                f"Internal Texture: {row['Internal Texture']}",
            ])

        raw = self.tokenizer(
            flat_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        n_classes = len(df)
        reshaped = {
            k: v.to(self.device).view(n_classes, 6, -1)
            for k, v in raw.items()
        }
        return reshaped

    # ==========================================
    #       Bi-fusion per stage
    # ==========================================

    def context_layer(self, idx: int, x: torch.Tensor, text_embeddings: torch.Tensor):
        """
        x: [B, C, H, W] or [B, C, D, H, W]
        text_embeddings: [num_channels, seq_len, d_model]
        Returns:
          x_out: same spatial shape as x
          text_out: [num_channels, seq_len, d_model]
        """
        if x.ndim == 4:
            b, c, h, w = x.shape
            depth = 1
        elif x.ndim == 5:
            b, c, depth, h, w = x.shape
        else:
            raise ValueError(f"Unexpected x.ndim={x.ndim} in context_layer")

        n_ch, seq_len, d_model = text_embeddings.shape

        # Expand text for batch, flatten channels*tokens
        txt = text_embeddings.unsqueeze(0).expand(b, -1, -1, -1)   # [B, n_ch, seq_len, d]
        txt = txt.reshape(b, n_ch * seq_len, d_model)              # [B, n_ch*seq_len, d]
        txt = F.normalize(txt, p=2, dim=-1)

        # Visual flattening: preserve your modified 2D behavior (x_flat = x)
        if x.ndim == 4:
            vis = x
        else:
            vis = x.flatten(2).transpose(1, 2)                     # [B, D*H*W, C]

        # Optional channel projection (same semantics, new wiring)
        if idx < 4:
            layer = self.cp_layers[idx]
            rev   = self.reverse_cp_layers[idx]

            if txt.dtype == torch.float16:
                self.cp_layers[idx] = layer.to(self.device).half()
                self.reverse_cp_layers[idx] = rev.to(self.device).half()
            else:
                self.cp_layers[idx] = layer.to(self.device)
                self.reverse_cp_layers[idx] = rev.to(self.device)

            txt = self.cp_layers[idx](txt)

        # Bi-fusion: txt, vis are updated in-place
        txt, vis = self.contexts[idx](vis, None, txt)

        if idx < 4:
            txt = self.reverse_cp_layers[idx](txt)

        # Restore visual tensor shape
        if x.ndim == 4:
            vis_out = vis.transpose(1, 2).view(b, c, h, w)
        else:
            vis_out = vis.transpose(1, 2).view(b, c, depth, h, w)

        # Restore text shape: [n_ch, seq_len, d_model]
        txt_out = txt.view(b, n_ch, seq_len, d_model)[0]

        return vis_out, txt_out

    def encode_descriptions(self, descriptions):
        """
        Encode a batch of free-text descriptions.

        descriptions: list[str] length B
        returns: [B, L, D] last-layer hidden states
        """
        if isinstance(descriptions, (list, tuple)):
            text_list = list(descriptions)
        else:
            text_list = [str(descriptions)]

        tokens = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.text_encoder(0, **tokens)  # or self.text_encoder(**tokens) if vanilla HF
        return outputs.hidden_states[-1]
        
    # ==========================================
    #       Forward
    # ==========================================

    def forward(
        self,
        x: torch.Tensor,
        class_definitions=None,
        image_descriptions=None,
        device: str = "cuda",
    ):
        """
        x: [B, C, H, W]
        class_definitions:
            - str path or DataFrame (class-level priors, optional)
        image_descriptions:
            - list[str] of length B with 2–3 sentence descriptions per image
        returns:
            feats            : [B, out_channels]
            class_text_feats: list of attribute-level text hidden states (or None)
            logits          : [B, out_channels, H, W]
            img_text_emb    : [B, L, D] (per-image text embeddings) or None
        """
        self.device = device

        # -------------------------------
        # 1) Optional class-level text
        # -------------------------------
        class_text_feats = None
        txt_feat = None
        sent_tokens = None

        if class_definitions is not None:
            if isinstance(class_definitions, str):
                df = pd.read_csv(class_definitions)
            elif isinstance(class_definitions, pd.DataFrame):
                df = class_definitions
            else:
                raise ValueError("class_definitions must be a path or DataFrame")

            feat_tokens = self.QueryEmbedding(df)
            sent_tokens = self.TextEmbedding(df)
            num_attr = sent_tokens["input_ids"].shape[1]  # 6
            class_text_feats = [None] * num_attr

        swin_pyramid = []
        x_in = x

        # -------------------------------
        # 2) Swin + optional class text
        # -------------------------------
        for stage_idx in range(self.num_stages + 1):
            x, stage_feats = self.swinViT(stage_idx, x)
            swin_pyramid.append(stage_feats)

            if class_definitions is None:
                continue  # no class-text stream, just vision

            if stage_idx == 0:
                # feature-level text stream
                txt_feat = self.text_encoder(stage_idx, **feat_tokens).hidden_states[-1]

                # attribute-level streams for each of the 6 attributes
                for a in range(num_attr):
                    token_slice = {k: v[:, a, :] for k, v in sent_tokens.items()}
                    attr_out = self.text_encoder(stage_idx, **token_slice).hidden_states[-1]
                    class_text_feats[a] = attr_out
            else:
                # propagate feature-level text & fuse with vision
                txt_feat = self.text_encoder(stage_idx, txt_feat).hidden_states[-1]
                x, txt_feat = self.context_layer(stage_idx, x, txt_feat)

                # propagate attribute-level streams
                for a in range(num_attr):
                    attr_out = self.text_encoder(stage_idx, class_text_feats[a]).hidden_states[-1]
                    class_text_feats[a] = attr_out

        # -------------------------------
        # 3) UNETR encoder–decoder
        # -------------------------------
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(swin_pyramid[0])
        enc2 = self.encoder3(swin_pyramid[1])
        enc3 = self.encoder4(swin_pyramid[2])

        deep = self.encoder10(swin_pyramid[4])
        deep = self.deep_se(deep)

        feats = self.cp_feats(deep)  # [B, out_channels]

        d3 = self.decoder5(deep, swin_pyramid[3])
        d2 = self.decoder4(d3, enc3)
        d1 = self.decoder3(d2, enc2)
        d0 = self.decoder2(d1, enc1)
        dec_out = self.decoder1(d0, enc0)

        dec_out = self.final_refine(dec_out)
        logits = self.out(dec_out)   # [B, out_channels, H, W]

        # -------------------------------
        # 4) Per-image long descriptions
        # -------------------------------
        img_text_emb = None
        if image_descriptions is not None:
            img_text_emb = self.encode_descriptions(image_descriptions)  # [B, L, D]

        return feats, class_text_feats, logits, img_text_emb



