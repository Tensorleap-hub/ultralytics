#!/usr/bin/env python
"""Build a per-image aggressor-family metadata map.

Produces a JSON file mapping each image STEM to:
    {"family": <str>, "axis": <str>, "role": <str>}
where role is one of {"aggressor", "false_aggressor", "non_aggressor"}.

Inputs are produced by the dataset build and are read at runtime:
  Dataset root (--root):
    <root>/fam_<family>_val.txt        -> role "aggressor"
    <root>/fam_<family>_false_val.txt  -> role "false_aggressor"
    <root>/non_val_aggressors.txt      -> role "non_aggressor" (family/axis "none")
      Each line looks like "./images/val/<stem>.jpg"; stem = Path(line).stem.
  Resolved train JSONs (--resolved):
    <resolved>/<family>__train.json    -> JSON list of COCO paths like
      "train2017/000000123.jpg"; stem = path.replace("/", "_").rsplit(".", 1)[0];
      role "aggressor".

If a stem is seen more than once, the FIRST assignment wins (aggressor
precedence) and is not overwritten. Missing input files are skipped with a
warning. Output JSON is compact (no indent).

Run as:
    python scripts/build_aggressor_map.py
"""

import argparse
import json
from pathlib import Path

# Aggressor family -> axis mapping. Iteration order over families is the
# insertion order of this dict.
FAMILY_AXIS = {
    "cats": "lightness",
    "cakes": "size",
    "spoons": "aspect",
    "airplanes_sky_vs_road": "context",
    "dogs_indoor_vs_outdoor": "context",
    "snowboards": "lowshot",
    "dominant_ovens_vs_microwaves": "confusion",
    "synthetic_bus": "contrast",
    "synthetic_zebra": "noise",
}

DEFAULT_ROOT = "/Users/yamtawachi/tensorleap/datasets/coco_subset_partA"
DEFAULT_RESOLVED = (
    "/Users/yamtawachi/tensorleap/aggressor-benchmarking/od_analysis/outputs/"
    "coco_aggressor_subdataset/resolved"
)
DEFAULT_OUT = "/Users/yamtawachi/tensorleap/ultralytics/aggressor_map.json"


def assign(mapping, stem, family, axis, role):
    """Record stem->info, keeping the FIRST assignment (no overwrite)."""
    if not stem or stem in mapping:
        return
    mapping[stem] = {"family": family, "axis": axis, "role": role}


def read_txt_stems(path):
    """Yield Path(line).stem for each non-empty line in a txt file."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Path(line).stem


def read_json_stems(path):
    """Yield resolved stems for each COCO path in a JSON list file."""
    with open(path, "r") as f:
        paths = json.load(f)
    for p in paths:
        p = str(p).strip()
        if not p:
            continue
        yield p.replace("/", "_").rsplit(".", 1)[0]


def build_map(root, resolved):
    root = Path(root)
    resolved = Path(resolved)
    mapping = {}

    # Aggressor precedence: process aggressor sources first so that any stem
    # later appearing as a false_aggressor / non_aggressor does not overwrite.

    # 1) Resolved train JSONs -> role "aggressor".
    for family in FAMILY_AXIS:
        axis = FAMILY_AXIS[family]
        json_path = resolved / f"{family}__train.json"
        if not json_path.exists():
            print(f"WARNING: missing resolved train JSON: {json_path}")
            continue
        for stem in read_json_stems(json_path):
            assign(mapping, stem, family, axis, "aggressor")

    # 2) Per-family val aggressors -> role "aggressor".
    for family in FAMILY_AXIS:
        axis = FAMILY_AXIS[family]
        val_path = root / f"fam_{family}_val.txt"
        if not val_path.exists():
            print(f"WARNING: missing aggressor val list: {val_path}")
            continue
        for stem in read_txt_stems(val_path):
            assign(mapping, stem, family, axis, "aggressor")

    # 3) Per-family false val aggressors -> role "false_aggressor".
    for family in FAMILY_AXIS:
        axis = FAMILY_AXIS[family]
        false_path = root / f"fam_{family}_false_val.txt"
        if not false_path.exists():
            print(f"WARNING: missing false aggressor val list: {false_path}")
            continue
        for stem in read_txt_stems(false_path):
            assign(mapping, stem, family, axis, "false_aggressor")

    # 4) Non val aggressors -> role "non_aggressor" (family/axis "none").
    non_path = root / "non_val_aggressors.txt"
    if not non_path.exists():
        print(f"WARNING: missing non-aggressor val list: {non_path}")
    else:
        for stem in read_txt_stems(non_path):
            assign(mapping, stem, "none", "none", "non_aggressor")

    return mapping


def print_counts(mapping):
    by_role = {}
    by_family = {}
    for info in mapping.values():
        by_role[info["role"]] = by_role.get(info["role"], 0) + 1
        by_family[info["family"]] = by_family.get(info["family"], 0) + 1

    print(f"\nTotal stems: {len(mapping)}")

    print("\nBy role:")
    for role in sorted(by_role):
        print(f"  {role}: {by_role[role]}")

    print("\nBy family:")
    for family in sorted(by_family):
        print(f"  {family}: {by_family[family]}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a per-image aggressor-family metadata map."
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="Dataset root containing fam_<family>_val.txt / "
        "fam_<family>_false_val.txt / non_val_aggressors.txt "
        f"(default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--resolved",
        default=DEFAULT_RESOLVED,
        help="Directory containing <family>__train.json files "
        f"(default: {DEFAULT_RESOLVED})",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output JSON path (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    mapping = build_map(args.root, args.resolved)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(mapping, f, separators=(",", ":"))
    print(f"\nWrote {len(mapping)} entries to {out_path}")

    print_counts(mapping)


if __name__ == "__main__":
    main()
