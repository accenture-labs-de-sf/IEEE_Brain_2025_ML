"""Full EEGPT (encoder + predictor + reconstructor) for the **decoded-signal** route.

Loads the original EEGPT pretraining checkpoint (the ``large4E`` variant) into the vendored,
torch-only model and runs **masked reconstruction** — so we can run our ERDS analysis on
model-*reconstructed* EEG, not just latents. This complements the braindecode *encoder* path
(`eegxai.models.eegpt`, features only).

Model contract (note: **different** from the braindecode encoder path):
  * **58 channels** (:data:`USE_CHANNELS`), **256 Hz**, **4 s = 1024 samples**, ``patch_size=64``.
  * checkpoint: ``data/external/eegpt_mcae_58chs_4s_large4E.ckpt`` (~1 GB, from Figshare,
    git-ignored). See ``docs/eegpt-reconstruction.md`` for how to obtain it.

Reconstruction follows EEGPT pretraining: ``encoder(context) -> predictor -> reconstructor``.
The model code is vendored under ``vendor/`` (Apache-2.0; see ``vendor/README.md``).
"""

from __future__ import annotations

import random
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

from .vendor.eegpt_pretraining import (
    EEGTransformer, EEGTransformerPredictor, EEGTransformerReconstructor,
)

# 58 channels used during EEGPT pretraining (order matters; fed by name).
USE_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1",
    "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO3", "POZ", "PO4",
    "PO8", "O1", "OZ", "O2",
]
DEFAULT_CKPT = "data/external/eegpt_mcae_58chs_4s_large4E.ckpt"
SFREQ = 256.0
WINDOW_SAMPLES = 1024   # 4 s @ 256 Hz
PATCH_SIZE = 64
_CFG = dict(embed_dim=512, embed_num=4, depth=(8, 8, 8), num_heads=8)  # 'large' variant


def build_full_model(ckpt_path: str | Path = DEFAULT_CKPT, *, device: str = "cpu") -> dict:
    """Instantiate encoder+predictor+reconstructor and load the checkpoint. Returns a dict
    with the three modules plus ``chans_id`` and ``num_patches``."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"EEGPT checkpoint not found at {ckpt_path}. Download the ~1 GB "
            "eegpt_mcae_58chs_4s_large4E.ckpt from Figshare and place it there — see "
            "docs/eegpt-reconstruction.md."
        )
    nl = partial(nn.LayerNorm, eps=1e-6)
    ed, en, dep, nh = _CFG["embed_dim"], _CFG["embed_num"], _CFG["depth"], _CFG["num_heads"]
    common = dict(mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                  init_std=0.02, qkv_bias=True, norm_layer=nl)
    enc = EEGTransformer(img_size=[58, WINDOW_SAMPLES], patch_size=PATCH_SIZE,
                         embed_dim=ed, embed_num=en, depth=dep[0], num_heads=nh, **common)
    pred = EEGTransformerPredictor(num_patches=enc.num_patches, use_part_pred=True,
                                   embed_dim=ed, embed_num=en, predictor_embed_dim=ed,
                                   depth=dep[1], num_heads=nh, **common)
    rec = EEGTransformerReconstructor(num_patches=enc.num_patches, patch_size=PATCH_SIZE,
                                      embed_dim=ed, embed_num=en, reconstructor_embed_dim=ed,
                                      depth=dep[2], num_heads=nh, **common)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    for mod, pfx in [(enc, "encoder"), (pred, "predictor"), (rec, "reconstructor")]:
        mod.load_state_dict({k[len(pfx) + 1:]: v for k, v in sd.items()
                             if k.startswith(pfx + ".")}, strict=False)
        mod.to(device).eval()
    return {"encoder": enc, "predictor": pred, "reconstructor": rec,
            "chans_id": enc.prepare_chan_ids(USE_CHANNELS).to(device),
            "num_patches": enc.num_patches}


def make_masks(num_patches, *, mC_x: int = 12, p_n_y: float = 0.5, p_c_y: float = 0.2,
               seed: int | None = None):
    """Replicates EEGPT's pretraining mask (context ``mask_x`` / target ``mask_y``)."""
    if seed is not None:
        random.seed(seed); torch.manual_seed(seed)
    C, N = num_patches
    while True:
        mx, my, myb = [], [], []
        for i in range(N):
            ci = torch.randperm(C) + i * C
            if random.random() > p_n_y:
                mx.append(ci[:mC_x]); myb.append(ci[mC_x:])
            else:
                my.append(ci)
        if not mx or not myb:
            continue
        myb = torch.cat(myb, 0)
        myb = myb[torch.rand(myb.shape) < p_c_y]
        if len(myb) == 0:
            continue
        return torch.stack(mx, 0), torch.cat(my + [myb], 0)


@torch.no_grad()
def reconstruct(models: dict, x: torch.Tensor, *, mask=None):
    """``x``: ``(B, 58, 1024)``. Returns ``(reconstructed_patches, mask_x, mask_y)`` where
    reconstructed_patches is ``(B, n_masked, patch_size)`` — decoded signal for masked patches."""
    enc, pred, rec = models["encoder"], models["predictor"], models["reconstructor"]
    cid = models["chans_id"].to(x)
    mx, my = mask if mask is not None else make_masks(models["num_patches"])
    z = enc(x, cid, mask_x=mx)
    z, comb_z = pred(z, mask_x=mx)
    r = rec(comb_z, cid, mask_y=my)
    return r, mx, my
