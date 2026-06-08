# Dataset & Model — coco_subset_v4 + scratch-4 (aggressor benchmark)

What the Tensorleap integration in this repo runs on: the **V4 sub-COCO aggressor
benchmark** dataset and the **from-scratch YOLO11n** detector trained on it. This
file consolidates the dataset design, the aggressor taxonomy, the model, and the
metrics — pulled from the code in this repo and from the saved analysis in the
`aggressor-benchmarking` repo (branch `dataset-v4-600mains`).

> **Sources of truth** (don't paraphrase — read these if a number here looks off):
> - This repo: `ultralytics/cfg/datasets/coco_subset_v4.yaml`, `leap_binder.py`,
>   `scripts/build_aggressor_map.py`, `aggressor_map.json`, `leap_integration.py`.
> - `aggressor-benchmarking` (branch `dataset-v4-600mains`):
>   `od_analysis/coco_aggressor_subdataset/DATASET_V4.md` (dataset),
>   `…/TRAINING_YOLO11N.md` (training protocol),
>   `…/results/instance_analysis_v4/{eval_instance_summary,worst30_aggressor_analysis}.md`
>   (the scratch-4 metrics quoted below).

---

## TL;DR

- **Task:** 20-class object detection (YOLO11n) on a curated COCO subset.
- **The benchmark idea:** 9 **aggressor families** are baked into the data — each
  is a controlled *distribution shift* on one axis (lightness, size, context,
  noise, …). The model is trained to be a good detector but is **not** allowed to
  become robust to these shifts (augmentation that touches an aggressor axis is
  disabled). The benchmark then measures **how much worse** the model does on the
  shifted (aggressor) instances vs clean ones.
- **Three sample roles:** `aggressor` (the shifted main instances),
  `false_aggressor` (same family, clean — the within-family control),
  `non_aggressor` / `clean` (out-of-family clean instances — the global baseline).
- **Headline result (scratch-4):** true-aggressor instances score **0.271** mean
  per-box mAP50-95 vs **0.760** for the non-aggressor baseline — a **64% drop**.
  Aggressor instances are **15.7×** over-represented in the worst-30% of the val set.

---

## 1. The dataset: `coco_subset_v4`

Config: `ultralytics/cfg/datasets/coco_subset_v4.yaml` → data root
`/Users/yamtawachi/tensorleap/datasets/coco_subset_v4`.

### Classes (`nc=20`)

| idx | name | role in benchmark |
|---:|---|---|
| 0 | cat | aggressor main (lightness) |
| 1 | cake | aggressor main (size) |
| 2 | spoon | aggressor main (aspect) |
| 3 | airplane | aggressor main (context) |
| 4 | snowboard | aggressor main (lowshot) |
| 5 | **oven_like** | aggressor main (confusion) — **merged oven+microwave** |
| 6 | dog | aggressor main (context) |
| 7 | bus | aggressor main (noise, synthetic) |
| 8 | pizza | aggressor main (low_resolution, synthetic) |
| 9 | person | non-aggressor |
| 10 | train | non-aggressor |
| 11 | dining table | non-aggressor |
| 12 | giraffe | non-aggressor |
| 13 | bed | non-aggressor |
| 14 | toilet | non-aggressor |
| 15 | elephant | non-aggressor |
| 16 | horse | non-aggressor |
| 17 | truck | non-aggressor |
| 18 | motorcycle | non-aggressor |
| 19 | bird | non-aggressor |

`oven_like` (idx 5) is the **confusion-pair** family: oven + microwave collapsed
into one trained class. In V4 this is **native** — the model emits 20 classes with
oven_like already merged, so `leap_binder.py` keeps `EVAL_CLASS_MERGE = {}` (no
runtime canonicalization needed). The benchmark's instance-eval doc labels this
"19 classes" because it counts the oven+microwave source pair as the single merged
class — same thing, different counting.

### Splits

| list | what | size |
|---|---|---|
| `train_100.txt` | seeded 100-image train subset (the integration's train split) | 100 |
| `val_all.txt` | full validation pool | **11,409 imgs / 20,769 boxes** |

> `coco_subset_v4.yaml` points `train:` at `train_100.txt` (a small seeded slice
> for the Tensorleap session). The **model** was trained on the full V4 train set
> (~14,487 imgs) on EC2 — see §3. The integration loads the pre-trained weights;
> it does not retrain on the 100.

### Validation composition (the 20,769 boxes)

- **4,297 true-aggressor imgs** → **5,400 aggressor MAIN instances** (exactly
  600 per family × 9). This is the **26.0%** instance-share target (5,400 / 20,769).
- **343 false-aggressor instances** (single-object, same families, clean).
- **6,769 non-aggressor instances** (single-object, clean, out-of-family).
- The remainder is **context** (non-main boxes inside aggressor images): 8,257 boxes.

> **Config flag — `tensorleap_use_false_aggressors`** (`cfg/default.yaml`, currently
> `False`): whether the 343 false-aggressor images are kept in the val set. `True`
> keeps them (val = 11,409); `False` drops them (val = 11,066). They're the
> within-family clean control — keep them in to read `drop% vs false`, drop them for a
> leaner clean-baseline-only comparison.

Per-family validation (from `DATASET_V4.md`; `imgs` = val images, `mains` = main
instances, `ctx` = context boxes, `main-area med` = median normalized box area):

| family (axis) | val imgs | mains | singles | ctx | main-area med |
|---|---:|---:|---:|---:|---:|
| cats (lightness) | 560 | 600 | 522 | 257 | 0.469 |
| cakes (size) | 412 | 600 | 263 | 1903 | 0.011 |
| spoons (aspect) | 482 | 600 | 389 | 1003 | 0.017 |
| airplane (context) | 418 | 600 | 288 | 1025 | 0.493 |
| snowboards (lowshot) | 473 | 600 | 371 | 1566 | 0.025 |
| oven_like (confusion) | 401 | 600 | 227 | 233 | 0.219 |
| dogs (context) | 559 | 600 | 523 | 458 | 0.321 |
| synthetic_bus (noise σ50) | 463 | 600 | 356 | 1712 | 0.396 |
| synthetic_pizza (low-res 8×) | 529 | 600 | 469 | 100 | 0.797 |

### Train composition (full V4 train, ~14,487 imgs)

OBVIOUS aggressors only (main-area floor ≥ 0.05 + bottom-20% coherence prune;
spoons/snowboards exempt as inherently small):

| family | train imgs |
|---|---:|
| cats | 500 |
| cakes | 770 |
| spoons | 604 |
| airplane | 684 |
| snowboards | 45 |
| oven_like | 583 |
| dogs | 720 |
| bus | 792 |
| pizza | 789 |
| **aggressor subtotal** | **5,487** |
| non-aggressor (clutter-first, ≤2 boxes) | 9,000 |
| **total** | **14,487** |

---

## 2. Aggressors

An **aggressor** is a curated set of instances exhibiting one controlled
distribution shift. The shift (the "axis") is what the benchmark measures
robustness to. Defined in `scripts/build_aggressor_map.py`.

### Families → axes

| family | axis | nature |
|---|---|---|
| `cats` | lightness | natural (dark/low-light cats) |
| `cakes` | size | natural (small cakes; the prominent cakes live in train) |
| `spoons` | aspect | natural (thin/elongated) |
| `airplanes_sky_vs_road` | context | natural (background shift: sky vs ground) |
| `dogs_indoor_vs_outdoor` | context | natural (indoor vs outdoor) |
| `snowboards` | lowshot | natural (few-shot class — starved in train *by design*) |
| `dominant_ovens_vs_microwaves` | confusion | natural (oven↔microwave confusion pair) |
| `synthetic_bus` | noise | **synthetic** (Gaussian noise σ50 applied) |
| `synthetic_pizza` | low_resolution | **synthetic** (8× downscale/upscale) |

### Roles (per image, first-match wins; see `build_aggressor_map.py`)

| role | meaning | where it comes from |
|---|---|---|
| `aggressor` | the shifted main instances (train + val) | resolved train JSONs + `fam_<family>_val.txt` |
| `false_aggressor` | same family, **clean** — within-family control that isolates the shift from class difficulty | `fam_<family>_false_val.txt` |
| `non_aggressor` | clean, out-of-family — the global baseline | `non_val_aggressors.txt` (family/axis = `none`) |
| `clean` | default for anything not in the map (e.g. the 9,000 train non-aggressors) | runtime fallback in `metadata_aggressor` |

**Train vs val family construction.** Both train and val contribute `aggressor`
images for each family, but they're built differently:
- **Val** mains are the strict, coherence-ranked shift (600/family, the thing being
  measured). V4 raised the per-family target to 600 ("T=600 mains") via tiered
  expansion — fill tier-1 (original resolved val pool) first, top up from tier-2
  (same-criteria candidates excluded from every other split).
- **Train** aggressors are looser ("OBVIOUS aggressors only", area-floored) so the
  detector still learns the class — but augmentation on the aggressor axes is
  **off**, so it never becomes robust to the val shift.
- **Asymmetry to know:** `cakes` is small-cake by design on the val side (its
  prominent cakes are the *train* side) — so its false pool is large-cake and
  `drop% vs false` overstates cakes; read cakes against the non-aggressor baseline.

### `aggressor_map.json` (runtime map used by `leap_binder.py`)

Built by `build_aggressor_map.py`; maps each image **stem** →
`{family, axis, role}`. Current counts (`aggressor_map.json` in this repo):

- **16,896 stems total** — `aggressor` 9,784 · `false_aggressor` 343 · `non_aggressor` 6,769.
- `aggressor` = 5,487 train + 4,297 val. The 9,000 train non-aggressors are **not**
  in the map (they resolve to role `clean` at runtime).
- By family (train+val+false): cats 1,098 · cakes 1,219 · spoons 1,093 ·
  airplanes 1,175 · dogs 1,331 · snowboards 518 · oven_like 1,017 ·
  synthetic_bus 1,274 · synthetic_pizza 1,402 · none 6,769.

Rebuild: `python scripts/build_aggressor_map.py`.

---

## 3. The model: `scratch-4` (YOLO11n, from scratch)

- **Arch:** YOLO11n (nano). Trained **from random init** (no pretrained weights) as
  the detector-under-test.
- **Integrated weights:** `/Users/yamtawachi/tensorleap/datasets/models/subcoco_v4/yolo11n.onnx`
  (= `runs/detect/yolo11n_subcoco_scratch-4/weights/best.pt`, re-exported to ONNX
  with the train head so loss is computable). Loaded in `leap_integration.py`.
- **Training protocol** (`TRAINING_YOLO11N.md`): Ultralytics COCO recipe, imgsz 640,
  SGD lr0=0.01, with **aggressor-preserving augmentation** — `hsv=0`, `erasing=0`,
  `scale≤0.2`, `mosaic≤0.5`. This is the crux: full COCO augmentation would make the
  model robust to the very shifts (lightness/size/context) the benchmark measures
  and invalidate the result.
- **Model selection:** during training, validation runs on the **clean val only**
  (non-aggressor + false), so early-stopping is never done on the shifted
  distribution. The aggressor split is held out for the post-train eval.

> **Version drift to know:** `DATASET_V4.md` lists `scratch-7` (300 ep) as the
> *next* retrain target. The model currently wired into the Tensorleap integration
> is **scratch-4** (the metrics in §4 are scratch-4 on the V4 val set). The older
> `results/eval/eval_summary.{md,json}` and `results/eval/split_counts.json` in the
> benchmarking repo are **stale** (scratch-2, with `zebra`/separate `microwave` and
> a 14,300-img val) — ignore them for V4; use `results/instance_analysis_v4/`.

---

## 4. Metrics

### 4a. What the Tensorleap integration computes (`leap_binder.py`)

**Per-sample metrics** — every one is sliceable by the aggressor metadata below:

| metric | direction | meaning |
|---|---|---|
| `ious` | ↑ Up | mean IoU of greedy one-to-one matched preds vs GT (plus per-class IoU for the configured `wanted_cls_dic`) |
| `cost` | ↓ Down | YOLO loss parts: `box`, `cls`, `dfl` |
| `total_loss` (custom loss) | — | combined detection loss |
| `Confusion Matrix` | ↓ Down | per-prediction TP/FP/FN with confidence (IoU≥`cfg.iou`, oven_like merged) |

**Per-instance metrics** (one value per GT box — for element-instance analysis):

| metric | direction | meaning |
|---|---|---|
| `instance_best_iou` | ↓ Down | localization **error** `1 − IoU` of the best same-class prediction for this GT box (**1** if unmatched; lower is better). Registered name kept for continuity, but it now holds `1 − IoU`, not IoU. |
| `instance_match_confidence` | ↑ Up | confidence of that best-matching prediction (0 if unmatched) |

**Aggressor metadata** (`metadata_aggressor`) attached to every sample — this is the
key used to stratify all of the above:

```
aggressor_family : "cats" | … | "synthetic_pizza" | "none"
aggressor_axis   : "lightness" | … | "low_resolution" | "none"
aggressor_role   : "aggressor" | "false_aggressor" | "non_aggressor" | "clean"
is_aggressor     : 1 if role=="aggressor" else 0
```

Plus per-image stats (`metadata_per_img`): image path, #classes, #objects, bbox
area mean/var/median/min/max, and occlusion (bbox overlap / max overlap).

### 4b. Benchmark results — scratch-4 on V4 val

Instance-level per-box mAP50-95 (conf 0.001), from
`results/instance_analysis_v4/eval_instance_summary.md`.

**Baseline** (non-true-aggressors = non_aggressor 9k-pool + false): 7,112 instances,
mean **0.7597**.

**Pool summary:**

| pool | n_inst | mean mAP50-95 | drop% vs baseline |
|---|---:|---:|---:|
| true_aggressors_all | 5,400 | **0.2712** | **64.3%** |
| non_true_aggressors_baseline | 7,112 | 0.7597 | 0.0 |
| non_val_aggressors (9k only) | 6,769 | 0.7606 | −0.1 |
| false_aggressors_all | 343 | 0.7420 | 2.3 |

**Per-family true aggressors** (sorted worst→best; `drop% vs false` isolates the
perturbation from class difficulty):

| family (axis) | mean mAP50-95 | F1 | drop% vs baseline | drop% vs false |
|---|---:|---:|---:|---:|
| spoons (aspect) | 0.0023 | 0.00 | 99.7 | 99.1 |
| cakes (size) | 0.0168 | 0.00 | 97.8 | 97.2 |
| snowboards (lowshot) | 0.0555 | 0.01 | 92.7 | — |
| synthetic_bus (noise) | 0.2010 | 0.26 | 73.5 | 78.2 |
| dogs (context) | 0.3352 | 0.45 | 55.9 | 56.3 |
| oven_like (confusion) | 0.3462 | 0.40 | 54.4 | 43.7 |
| cats (lightness) | 0.3567 | 0.49 | 53.0 | 41.3 |
| airplane (context) | 0.4978 | 0.71 | 34.5 | 39.6 |
| synthetic_pizza (low-res) | 0.6297 | 0.66 | 17.1 | 24.1 |

**Worst-30% concentration** (`worst30_aggressor_analysis.md`): of the 6,231
lowest-IoU instances (cutoff IoU ≤ 0.103), **45.6%** are aggressor mains (1.76× their
26% base rate). **52.7%** of all aggressor mains fall in the worst-30% vs only
**3.3%** of non-aggressors → **15.7×** enrichment. Per-family capture rate:
cakes 92.7% · spoons 87.8% · synthetic_bus 72.0% · snowboards 64.7% · dogs 47.5% ·
cats 41.3% · oven_like 28.7% · pizza 22.0% · airplane 17.3%.

---

## 5. Where it lives (Tensorleap integration)

| file | role |
|---|---|
| `ultralytics/cfg/datasets/coco_subset_v4.yaml` | dataset paths, classes, splits |
| `leap_binder.py` | encoders, `metadata_aggressor`, per-img stats, all metrics & visualizers |
| `aggressor_map.json` | stem → {family, axis, role}, loaded at import |
| `scripts/build_aggressor_map.py` | regenerates the map from the dataset's split lists |
| `leap_integration.py` | loads `subcoco_v4/yolo11n.onnx` (scratch-4) and runs the binder end-to-end |
| `scripts/smoke_partA.py` | smoke test of the pipeline (oven_like / synthetic families) |

---

_Generated 2026-06-08 from this repo's code + the `aggressor-benchmarking`
`dataset-v4-600mains` analysis. If you retrain (scratch-7+) or rebuild the dataset,
update §3–§4 from the new `results/instance_analysis_*` outputs._
