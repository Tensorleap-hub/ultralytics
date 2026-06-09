# Download the coco_subset data from S3 and point this repo at it

Fetch a packaged `coco_subset` dataset from S3, extract it where the repo expects,
and switch the integration to it. **Three versions** exist — **v4**, **v4m**, **v4m2**
— identical except for the **non-aggressor *validation* images** (train, true-aggressor,
and false-aggressor pools are byte-identical across all three). Swap the version tag and
everything else is the same:

- **v4** — non-aggr largest-first, single-object (6,769 imgs, 1 box/img)
- **v4m** — non-aggr false-matched on size + brightness, single-object (6,769 imgs, 1 box/img)
- **v4m2** — non-aggr scene-matched to the true aggressors, multi-object (3,384 imgs, 2 box/img);
  newest — also removes the scene-complexity shift (see `DATASET_AND_MODEL.md` §5 for the
  metric comparison across all three).

The **active** version = whatever `data:` in `ultralytics/cfg/default.yaml` points at
(currently `coco_subset_v4m`).

> What the dataset/model/metrics actually are: see `DATASET_AND_MODEL.md`.

## 1. Versions on S3

Private bucket `aggrresors-benchmarking` (~4.1 GB each, contains `data.yaml`,
`images/{train,val}/`, `labels/{train,val}/`, the `*.txt` split lists, and
`split_counts.json`):

| version | S3 key (under `s3://aggrresors-benchmarking/public-datasets/`) | local data root |
|---|---|---|
| **v4m2** (newest) | `coco_subset_v4m2/coco_subcoco_dataset.tar.gz` | `<data_path>/coco_subset_v4m2` |
| v4m (currently wired) | `coco_subset_v4m/coco_subcoco_dataset.tar.gz` | `<data_path>/coco_subset_v4m` |
| v4 | `coco_subset_v4/coco_subcoco_dataset.tar.gz` | `<data_path>/coco_subset_v4` |

`<data_path>` = your local datasets directory (= `tensorleap_path` in
`ultralytics/cfg/default.yaml`, e.g. `/Users/<you>/tensorleap/datasets`).

## 2. Download + extract (where it goes)

```bash
export AWS_PROFILE=dev          # private bucket; run `aws sso login` if the session expired
VER=v4m                         # or: v4, v4m2

DST=<data_path>/coco_subset_$VER
mkdir -p "$DST"
aws s3 cp "s3://aggrresors-benchmarking/public-datasets/coco_subset_$VER/coco_subcoco_dataset.tar.gz" "$DST/"
tar -xzf "$DST/coco_subcoco_dataset.tar.gz" -C "$DST" && rm "$DST/coco_subcoco_dataset.tar.gz"

cat "$DST/split_counts.json"    # verify — v4/v4m: val_all 11409, non_val_aggressors 6769, val_clean 7112
                                #          v4m2:   val_all  8024, non_val_aggressors 3384, val_clean 3727
```

The data root must match the `path:` in the repo's dataset yaml (§3). Ignore the
`path:` inside the extracted `data.yaml` — the repo doesn't use it.

## 3. Point the repo at it (the only configs that matter)

1. **`ultralytics/cfg/default.yaml`** → `data: coco_subset_<VER>.yaml`
   (currently `coco_subset_v4m.yaml`). Model stays `model: models/subcoco_v4/yolo11n.pt`.
2. **`ultralytics/cfg/datasets/coco_subset_<VER>.yaml`** — already in the repo
   (`val: val_all.txt`, `nc: 20`). Set its `path:` to `<data_path>/coco_subset_<VER>`
   if your `<data_path>` differs from the value committed there.
3. **Regenerate `aggressor_map.json`** for the chosen root (each version's non-aggressor
   stems differ — the default `--root` is **v4**, so pass it explicitly otherwise):
   ```bash
   python scripts/build_aggressor_map.py --root <data_path>/coco_subset_v4m2
   # v4m:  python scripts/build_aggressor_map.py --root <data_path>/coco_subset_v4m
   # v4:   python scripts/build_aggressor_map.py     (default root = coco_subset_v4)
   ```

Optional knob — **`tensorleap_use_false_aggressors`** in `default.yaml` (currently
`False`): keep/drop the 343 false-aggressor images in the val set (v4/v4m: 11,409 ↔ 11,066;
v4m2: 8,024 ↔ 7,681).



Then run the integration from the repo root with this repo's `.venv`
(e.g. `python leap_integration.py`).

## 4. Push to the platform

Push the ONNX model + this integration code and kick off an Evaluate run
(`leapdev` `24.rc-0`):

```bash
leapdev push -m <data_path>/models/subcoco_v4/yolo11n.onnx --eval
```

## 5. View instances & aggressors in the analysis

Before starting the analysis, set these up to see the instance views and aggressors:

1. Toggle **Element Instance** on.
2. Filter `metadata.builtin_instance_metadata_is_instance == 1`.
3. Filter `dataset_state.keyword = valid`.
4. To see the aggressor family/role, **color by** `metadata.aggressor_aggressor_role`
   (role) or `metadata.aggressor_aggressor_family` (family).
5. Regenerate insights based on these filters.
