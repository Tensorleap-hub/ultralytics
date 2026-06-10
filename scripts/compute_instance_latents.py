"""Offline precompute of per-instance latent vectors for the Tensorleap integration.

Mirrors the zeitview pattern (precompute -> NPZ -> load at runtime), adapted to be
PER-INSTANCE for the element-instance binder:

  * Latent source = the LAST YOLO layer's features: the 3 feature maps that feed the
    Detect head (strides 8/16/32, 84 ch each = 20 cls + 64 DFL box logits). Captured
    with a forward-pre-hook on the Detect module.
  * For each GT box we ROI-pool (crop + spatial mean) on each of the 3 scales and
    concat -> a 252-d vector per instance.
  * All val instance vectors are standardized then reduced to 60-d with PCA (numpy SVD).
  * Saved to instance_latents_val.npz keyed by "<image_stem>__<instance_idx>", plus a
    per-sample mean keyed by "<image_stem>" (used for the base sample id's latent).

Only the VALIDATION split is computed (per the task: train latent can be all zeros).
The runtime `instance_latent_space` in leap_binder.py loads this NPZ and returns zeros
for any id not present (so train + missing -> zeros).

Run from the repo root with this repo's .venv:
    python scripts/compute_instance_latents.py
"""
import os
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from code_loader.contract.datasetclasses import PreprocessResponse, DataStateType

import leap_binder as lb
from ultralytics.tensorleap_folder.utils import create_data_with_ult, pre_process_dataloader

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "instance_latents_val.npz")
PCA_DIM = 60
IMG = 640  # binder preprocesses to 640x640


def _roi_pool(feature_maps, box_xyxy_px):
    """Crop each scale's [C,H,W] map at the box (in 640px coords), spatial-mean-pool,
    concat across scales -> 1-D vector. Clamps to >=1 cell so tiny boxes still pool."""
    vecs = []
    x1, y1, x2, y2 = box_xyxy_px
    for fm in feature_maps:                       # fm: [C, H, W]
        C, H, W = fm.shape
        sx, sy = IMG / W, IMG / H                  # stride for this scale
        fx1 = max(0, int(np.floor(x1 / sx)))
        fy1 = max(0, int(np.floor(y1 / sy)))
        fx2 = min(W, max(fx1 + 1, int(np.ceil(x2 / sx))))
        fy2 = min(H, max(fy1 + 1, int(np.ceil(y2 / sy))))
        vecs.append(fm[:, fy1:fy2, fx1:fx2].mean(axis=(1, 2)))
    return np.concatenate(vecs).astype(np.float32)  # [sum(C)]


def main():
    print(f"model: {lb.cfg.model} | data: {lb.cfg.data}")
    model = YOLO(str((Path(lb.cfg.tensorleap_path) / lb.cfg.model).resolve())).model.eval()
    detect = model.model[-1]
    captured = {}
    detect.register_forward_pre_hook(lambda m, inp: captured.__setitem__("f", inp[0]))

    # Build the FULL val set (exclude_stems=None) so latents exist for every val image
    # regardless of the false-aggressor flag; keyed by stem so the runtime lookup is
    # independent of split ordering / filtering.
    ds, n = create_data_with_ult(lb.cfg, lb.yolo_data, phase="val", exclude_stems=None)
    pr = PreprocessResponse(sample_ids=[str(i) for i in range(n)],
                            data={"dataloader": ds}, sample_id_type=str,
                            state=DataStateType.validation)

    keys, raw = [], []          # per-instance id + 252-d pooled vector
    sample_of = []              # parallel list: image stem each instance belongs to
    for k in range(n):
        imgs, clss, bboxes, _ = pre_process_dataloader(pr, k, lb.predictor)
        stem = Path(ds.im_files[k]).stem
        x = torch.from_numpy(imgs).float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            model(x)
        fmaps = [f[0].cpu().numpy() for f in captured["f"]]   # 3 x [C,H,W]
        for i, (box, c) in enumerate(zip(bboxes, clss.reshape(-1))):
            cx, cy, bw, bh = [float(v) for v in box]
            if not np.isfinite([cx, cy, bw, bh, float(c)]).all():
                continue
            x1 = (cx - bw / 2) * IMG; y1 = (cy - bh / 2) * IMG
            x2 = (cx + bw / 2) * IMG; y2 = (cy + bh / 2) * IMG
            keys.append(f"{stem}__{i}")
            sample_of.append(stem)
            raw.append(_roi_pool(fmaps, (x1, y1, x2, y2)))
        if (k + 1) % 1000 == 0:
            print(f"  {k+1}/{n} images, {len(raw)} instances")

    X = np.stack(raw).astype(np.float32)
    print(f"collected {X.shape[0]} instances, raw dim {X.shape[1]}")

    # standardize -> PCA (numpy SVD) -> PCA_DIM
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xs = (X - mu) / sd
    Xc = Xs - Xs.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:PCA_DIM]                       # [PCA_DIM, raw_dim]
    Z = (Xc @ comps.T).astype(np.float32)       # [N, PCA_DIM]
    print(f"PCA -> {Z.shape}; explained-var fraction "
          f"{float((np.var(Z,0).sum())/np.var(Xc,0).sum()):.3f}")

    out = {keys[j]: Z[j] for j in range(len(keys))}
    # per-sample mean latent (for the base sample id "<stem>")
    by_stem = {}
    for j, stem in enumerate(sample_of):
        by_stem.setdefault(stem, []).append(Z[j])
    for stem, vs in by_stem.items():
        out[stem] = np.mean(vs, axis=0).astype(np.float32)

    np.savez_compressed(OUT_PATH, **out)
    print(f"saved {len(out)} vectors ({len(keys)} instances + {len(by_stem)} sample-means) "
          f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
