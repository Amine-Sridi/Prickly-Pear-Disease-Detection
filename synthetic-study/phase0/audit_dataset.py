"""Phase 0 read-only audit of DatasetPreped / DatasetPreped2.

Usage: python -I audit_dataset.py <repo_root> <out_json>
Writes nothing inside the repository.
"""
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1])
out_json = Path(sys.argv[2])
D2 = root / "DatasetPreped2"
D1 = root / "DatasetPreped"
SPLITS = ["train", "val", "test"]
CLASSES = ["Healthy", "Diseased"]
CLS_ID = {"Healthy": 0, "Diseased": 1}
TILE_RE = re.compile(r"^(?P<src>.+)_(?P<y0>\d+)_(?P<y1>\d+)_(?P<x0>\d+)_(?P<x1>\d+)$")
IMG_EXT = {".png", ".jpg", ".jpeg"}

report = {}


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def dhash(img, size=8):
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def thumb(img, size=16):
    g = img.convert("RGB").resize((size, size), Image.BILINEAR)
    v = np.asarray(g, dtype=np.float32).flatten()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


records = []  # one per image in DatasetPreped2
for s in SPLITS:
    for c in CLASSES:
        for p in sorted((D2 / "images" / s / c).iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            m = TILE_RE.match(p.stem)
            with Image.open(p) as im:
                size = im.size
                mode = im.mode
                dh = dhash(im)
                th = thumb(im)
            lab = D2 / "labels" / s / c / (p.stem + ".txt")
            boxes = []
            if lab.exists():
                for line in lab.read_text().strip().splitlines():
                    parts = line.split()
                    if parts:
                        boxes.append([float(x) for x in parts])
            records.append(dict(
                split=s, cls=c, name=p.name, path=p, size=size, mode=mode,
                src=m["src"] if m else None,
                tile=(int(m["y0"]), int(m["y1"]), int(m["x0"]), int(m["x1"])) if m else None,
                md5=md5(p), dh=dh, th=th, has_label=lab.exists(), boxes=boxes,
            ))

# ---- counts, sizes, labels
counts = {}
for s in SPLITS:
    for c in CLASSES:
        rs = [r for r in records if r["split"] == s and r["cls"] == c]
        nbox = sum(len(r["boxes"]) for r in rs)
        counts[f"{s}/{c}"] = dict(
            images=len(rs),
            boxes=nbox,
            missing_label=sum(not r["has_label"] for r in rs),
            empty_label=sum(r["has_label"] and not r["boxes"] for r in rs),
            sizes=Counter(f"{r['size'][0]}x{r['size'][1]}" for r in rs).most_common(),
            modes=Counter(r["mode"] for r in rs).most_common(),
            class_ids=Counter(int(b[0]) for r in rs for b in r["boxes"]).most_common(),
            class_id_mismatch_folder=sum(1 for r in rs for b in r["boxes"] if int(b[0]) != CLS_ID[c]),
            distinct_boxes=Counter(tuple(b[1:]) for r in rs for b in r["boxes"]).most_common(5),
            boxes_per_image=Counter(len(r["boxes"]) for r in rs).most_common(),
            tile_px=Counter((r["tile"][1] - r["tile"][0], r["tile"][3] - r["tile"][2]) for r in rs if r["tile"]).most_common(),
            unparsed_names=sum(r["src"] is None for r in rs),
        )
report["counts"] = counts

# orphan label files
orphans = {}
for s in SPLITS:
    for c in CLASSES:
        imgs = {p.stem for p in (D2 / "images" / s / c).iterdir()}
        labs = {p.stem for p in (D2 / "labels" / s / c).iterdir() if p.suffix == ".txt"}
        orphans[f"{s}/{c}"] = len(labs - imgs)
report["orphan_labels"] = orphans

# ---- DatasetPreped vs DatasetPreped2 identity
same = diff = missing = 0
for r in records:
    q = D1 / r["split"] / r["cls"] / r["name"]
    if not q.exists():
        missing += 1
    elif md5(q) == r["md5"]:
        same += 1
    else:
        diff += 1
report["DatasetPreped_vs_2"] = dict(identical=same, different=diff, missing_in_DatasetPreped=missing)

# ---- source photos
src_split = defaultdict(set)
src_cls = defaultdict(set)
for r in records:
    src_split[r["src"]].add(r["split"])
    src_cls[r["src"]].add(r["cls"])
per_split_src = {s: {r["src"] for r in records if r["split"] == s} for s in SPLITS}
overlap = {}
for a, b in combinations(SPLITS, 2):
    overlap[f"{a}&{b}"] = len(per_split_src[a] & per_split_src[b])
report["source_photos"] = dict(
    unique_sources_total=len(src_split),
    unique_sources_per_split={s: len(v) for s, v in per_split_src.items()},
    sources_shared_between_splits=overlap,
    sources_in_all_three=len(per_split_src["train"] & per_split_src["val"] & per_split_src["test"]),
    sources_with_both_classes=sum(len(v) > 1 for v in src_cls.values()),
    name_families=Counter("img_YYYYMMDD_HHMMSS" if re.match(r"img_\d{8}_\d{6}$", k)
                          else "img-YYYYMMDD-waNNNN" if re.match(r"img-\d{8}-wa\d+$", k)
                          else "uuid" if re.match(r"[0-9a-f]{8}-", k) else "other"
                          for k in src_split).most_common(),
)
# fraction of val/test tiles whose source photo also has tiles in train
leak = {}
for s in ["val", "test"]:
    for c in CLASSES + ["all"]:
        rs = [r for r in records if r["split"] == s and (c == "all" or r["cls"] == c)]
        k = sum(r["src"] in per_split_src["train"] for r in rs)
        leak[f"{s}/{c}"] = dict(tiles=len(rs), source_also_in_train=k,
                                frac=round(k / len(rs), 4) if rs else None)
report["tile_level_leakage"] = leak

# capture dates of the timestamped photos per split
dates = defaultdict(Counter)
for r in records:
    m = re.match(r"img[_-](\d{8})", r["src"] or "")
    dates[r["split"]][m.group(1) if m else "no-date(uuid)"] += 1
report["capture_dates_tiles"] = {s: dict(v) for s, v in dates.items()}

# ---- exact duplicates (md5)
by_md5 = defaultdict(list)
for r in records:
    by_md5[r["md5"]].append(r)
dup_groups = [g for g in by_md5.values() if len(g) > 1]
report["exact_duplicates"] = dict(
    groups=len(dup_groups),
    cross_split_groups=sum(len({r["split"] for r in g}) > 1 for g in dup_groups),
    cross_class_groups=sum(len({r["cls"] for r in g}) > 1 for g in dup_groups),
    examples=[[f"{r['split']}/{r['cls']}/{r['name']}" for r in g] for g in dup_groups[:5]],
)

# ---- near duplicates between val/test and train (different source photo only)
train = [r for r in records if r["split"] == "train"]
T_dh = np.stack([r["dh"] for r in train])
T_th = np.stack([r["th"] for r in train])
near = {}
examples = []
for s in ["val", "test"]:
    rs = [r for r in records if r["split"] == s]
    stats = Counter()
    for r in rs:
        ham = (T_dh != r["dh"]).sum(1)
        cos = T_th @ r["th"]
        other = np.array([t["src"] != r["src"] for t in train])
        best_any = int(ham.min())
        cos_other = cos[other]
        ham_other = ham[other]
        stats["n"] += 1
        stats["dhash<=4_any"] += best_any <= 4
        stats["dhash<=4_other_src"] += int(ham_other.min()) <= 4 if other.any() else 0
        stats["thumbcos>=0.95_any"] += float(cos.max()) >= 0.95
        stats["thumbcos>=0.95_other_src"] += float(cos_other.max()) >= 0.95 if other.any() else 0
        if other.any() and float(cos_other.max()) >= 0.95 and len(examples) < 12:
            j = np.flatnonzero(other)[int(cos_other.argmax())]
            examples.append(dict(query=f"{s}/{r['cls']}/{r['name']}",
                                 train_match=f"train/{train[j]['cls']}/{train[j]['name']}",
                                 cos=round(float(cos_other.max()), 4), dhash=int(ham[j])))
    near[s] = dict(stats)
report["near_duplicates_vs_train"] = near
report["near_duplicate_examples"] = examples

# tile adjacency: val/test tiles that touch a train tile of the same source photo
adj = Counter()
train_tiles = defaultdict(list)
for r in train:
    train_tiles[r["src"]].append(r["tile"])
for r in records:
    if r["split"] == "train" or not r["tile"]:
        continue
    y0, y1, x0, x1 = r["tile"]
    touch = any(not (b[1] < y0 or b[0] > y1 or b[3] < x0 or b[2] > x1) for b in train_tiles.get(r["src"], []))
    adj[f"{r['split']}_touching_train_tile"] += touch
    adj[f"{r['split']}_total"] += 1
report["adjacent_tiles"] = dict(adj)

out_json.write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=2, default=str))
