"""Targeted smoke test for the Part A 20-class integration.

Builds the val dataloader directly (no full length-encoder sweep), finds known
sample types by stem, and exercises the full per-sample pipeline + assertions:
class alignment, aggressor metadata, instance length/mask encoders, all metrics.
"""
import numpy as np
from code_loader.contract.datasetclasses import (PreprocessResponse, DataStateType,
                                                  SamplePreprocessResponse)
from ultralytics.tensorleap_folder.global_params import cfg, yolo_data, all_clss
from ultralytics.tensorleap_folder.utils import create_data_with_ult
from leap_binder import (input_encoder, gt_encoder, metadata_aggressor, metadata_per_img,
                         instances_length_encoder, instance_mask_encoder,
                         loss, cost, ious, confusion_matrix_metric)
from leap_integration import load_model

print("building val dataloader (val_all.txt)...")
dl, n = create_data_with_ult(cfg, yolo_data, phase='val')
print(f"val n_samples = {n}")
pr = PreprocessResponse(sample_ids=[str(i) for i in range(n)],
                        data={'dataloader': dl}, sample_id_type=str,
                        state=DataStateType.validation)

# locate known sample types by stem substring
stems = [str(p) for p in dl.im_files]
def find(sub):
    for i, s in enumerate(stems):
        if sub in s:
            return i
    return None

import json as _json, os as _os
_amap = _json.load(open(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "aggressor_map.json")))
_non_stems = {k for k, v in _amap.items() if v.get("role") == "non_aggressor"}
def find_role(role_stems):
    from pathlib import Path as _P
    for i, s in enumerate(stems):
        if _P(s).stem in role_stems:
            return i
    return None

targets = {
    "synpizza": find("synpizza_"),
    "synbus": find("synbus_"),
    "non_aggr_example": find_role(_non_stems),
}
print("target indices:", targets)

model = load_model()
fails = []

def check(cond, msg):
    print(("  OK  " if cond else " FAIL ") + msg)
    if not cond:
        fails.append(msg)

# ---- find a val image with an oven_like GT (cls 5 in V4; natively merged) ----
mw_idx = None
for i in range(min(n, 4000)):
    g = gt_encoder(str(i), pr)
    if not np.isnan(g[:, 4]).all() and (g[:, 4].astype(int) == 5).any():
        mw_idx = i
        break
targets["oven_like_gt"] = mw_idx
print("oven_like-containing val idx:", mw_idx)

for name, idx in targets.items():
    if idx is None:
        check(False, f"[{name}] no sample found")
        continue
    sidx = str(idx)
    print(f"\n=== {name} (idx {idx}, stem {stems[idx]}) ===")
    img = input_encoder(sidx, pr)
    check(img.shape == (3, 640, 640), f"[{name}] per-sample image shape {img.shape}")
    img4 = img[None] if img.ndim == 3 else img  # framework batches at runtime; do it here
    gt = gt_encoder(sidx, pr)
    gt3 = gt[None] if gt.ndim == 2 else gt
    cls_ids = gt[:, 4]
    valid = cls_ids[~np.isnan(cls_ids)].astype(int)
    check(valid.size == 0 or (valid.min() >= 0 and valid.max() <= 19),
          f"[{name}] gt class ids in [0,19]: {sorted(set(valid.tolist()))}")

    y = model.run(None, {'images': img4})
    check(len(y) == 4 and y[0].shape[1] == 24, f"[{name}] model outputs={len(y)} head ch={y[0].shape[1]}")
    L = loss(y[1], y[2], y[3], gt3, y[0]); check(np.isfinite(L).all(), f"[{name}] loss={float(L[0]):.3f}")
    C = cost(y[1], y[2], y[3], gt3); check(all(np.isfinite(v).all() for v in C.values()), f"[{name}] cost ok")
    sp = SamplePreprocessResponse(np.array([idx]), pr)
    _ = ious(y[0], sp); _ = confusion_matrix_metric(y[0], sp)
    check(True, f"[{name}] ious + confusion_matrix ran")

    nlen = instances_length_encoder(sidx, pr)
    n_valid_gt = int((~np.isnan(gt[:, 4])).sum()) if not np.isnan(gt[:, 4]).all() else 0
    check(nlen == n_valid_gt, f"[{name}] instances_length={nlen} (gt boxes={n_valid_gt})")
    if nlen > 0:
        ei = instance_mask_encoder(sidx, pr, 0)
        check(ei is not None and ei.mask.shape == (3, 640, 640),
              f"[{name}] mask encoder -> label='{getattr(ei,'name',getattr(ei,'label','?'))}' mask{ei.mask.shape}")

    meta = metadata_aggressor(sidx, pr)
    print(f"  aggressor meta: {meta}")
    _ = metadata_per_img(sidx, pr)

# targeted metadata assertions
if targets["synpizza"] is not None:
    m = metadata_aggressor(str(targets["synpizza"]), pr)
    check(m["aggressor_family"] == "synthetic_pizza" and m["aggressor_axis"] == "low_resolution",
          f"[synpizza] family/axis = {m['aggressor_family']}/{m['aggressor_axis']}")
if targets["synbus"] is not None:
    m = metadata_aggressor(str(targets["synbus"]), pr)
    check(m["aggressor_family"] == "synthetic_bus" and m["aggressor_axis"] == "noise",
          f"[synbus] family/axis = {m['aggressor_family']}/{m['aggressor_axis']}")

print("\n==== SMOKE RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILURE(S): {fails}", "====")
