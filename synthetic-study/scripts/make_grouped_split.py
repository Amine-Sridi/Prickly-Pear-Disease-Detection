"""Build the grouped train/val/test split (evaluation option B).

A *group* joins every tile that could show the same physical scene:
  1. tiles cut from the same source photo (file-name prefix);
  2. source photos that are copies of each other (WhatsApp / UUID / downscaled
     copies), found as tile pairs with thumbnail cosine >= 0.97 whose crop
     coordinates match up to a scale factor of 1, 2 or 4;
  3. timestamped photos taken <= BURST_SECONDS (30 s) apart (same capture burst).
Groups are then assigned to train/val/test with a greedy, count-based balance
(Healthy first, then Diseased). No model output is used anywhere.

Run on the laptop (CPU only, ~1 min), from the repository root:
  python synthetic-study/scripts/make_grouped_split.py \
      --images DatasetPreped2/images --upstream-tree <upstream_tree.json> \
      --upstream-commit 730447f9bc1b05bbd58377c4fa271bc20583963e \
      --out synthetic-study/splits/split_v1.csv

The upstream tree JSON is the GitHub API response for
  https://api.github.com/repos/AnasBerka/CactiViT-materials/git/trees/<commit>?recursive=1
It is used to record each tile's upstream folder and git blob SHA, so Colab can
rebuild the dataset from the public repository and verify every file.
"""
import argparse
import datetime as dt
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test")
FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}
CLASSES = ("Healthy", "Diseased")
DISEASED_FOLDERS = ("Damaged", "EarlyStage", "LateStage", "Old_Dead")
TILE_RE = re.compile(r"^(?P<src>.+)_(\d+)_(\d+)_(\d+)_(\d+)\.png$")
TS_RE = re.compile(r"^img_(\d{8})_(\d{6})")
# 30 s, not 10 s: with 10 s, one cross-split pair of near-identical tiles remained
# (img_20210327_172718 / _172741, 23 s apart, cosine 0.95). 30 s leaves none.
BURST_SECONDS = 30
COPY_COS = 0.97
NEARDUP_AUDIT_COS = 0.95
SEED = 0


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def thumb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        v = np.asarray(im.convert("RGB").resize((16, 16), Image.BILINEAR), dtype=np.float32).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n else v


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="DatasetPreped2/images (old split folders)")
    ap.add_argument("--upstream-tree", required=True)
    ap.add_argument("--upstream-commit", required=True, help="commit SHA the tree was fetched for")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # upstream index: file name -> list of (folder, sha)
    tree = json.loads(Path(args.upstream_tree).read_text())
    if tree.get("truncated"):
        raise SystemExit("upstream tree listing is truncated")
    upstream_commit = args.upstream_commit
    up = defaultdict(list)
    for x in tree["tree"]:
        p = x["path"].split("/")
        if x["type"] == "blob" and len(p) == 4 and p[0] == "Datasets":
            up[p[3]].append((p[2], x["sha"]))

    # ---- collect tiles
    tiles = []
    for old_split in SPLITS:
        for cls in CLASSES:
            for path in sorted((Path(args.images) / old_split / cls).glob("*.png")):
                data = path.read_bytes()
                sha = git_blob_sha(data)
                match = [f for f, s in up.get(path.name, []) if s == sha]
                if len(match) != 1:
                    raise SystemExit(f"{path.name}: expected exactly one upstream match, got {match}")
                folder = match[0]
                if (cls == "Healthy") != (folder == "Healthy") or (cls == "Diseased" and folder not in DISEASED_FOLDERS):
                    raise SystemExit(f"{path.name}: class {cls} vs upstream folder {folder}")
                m = TILE_RE.match(path.name)
                coords = tuple(int(g) for g in m.groups()[1:])
                tiles.append(dict(file=path.name, cls=cls, upstream_folder=folder, blob_sha=sha,
                                  source=m["src"], coords=coords, old_split=old_split, path=path))
    print(f"tiles: {len(tiles)}")

    uf = UnionFind()
    for t in tiles:
        uf.find(t["source"])

    # ---- (2) copies of the same photo
    V = np.stack([thumb(t["path"]) for t in tiles])
    S = V @ V.T
    n_copy_links = 0
    for i, j in np.argwhere(np.triu(S, 1) >= COPY_COS):
        a, b = tiles[i], tiles[j]
        if a["source"] == b["source"]:
            continue
        ca, cb = np.array(a["coords"]), np.array(b["coords"])
        if any(np.array_equal(ca * k, cb) or np.array_equal(cb * k, ca) for k in (1, 2, 4)):
            if uf.find(a["source"]) != uf.find(b["source"]):
                n_copy_links += 1
            uf.union(a["source"], b["source"])
    # exact duplicates (md5-equal == blob-equal) are covered by the rule above, but enforce anyway
    by_sha = defaultdict(list)
    for t in tiles:
        by_sha[t["blob_sha"]].append(t["source"])
    for srcs in by_sha.values():
        for s in srcs[1:]:
            uf.union(srcs[0], s)

    # ---- (3) capture bursts
    stamps = {}
    for s in {t["source"] for t in tiles}:
        m = TS_RE.match(s)
        if m:
            stamps[s] = dt.datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S")
    order = sorted(stamps, key=stamps.get)
    for a, b in zip(order, order[1:]):
        if (stamps[b] - stamps[a]).total_seconds() <= BURST_SECONDS:
            uf.union(a, b)

    # ---- groups
    groups = defaultdict(lambda: Counter())
    for t in tiles:
        t["group"] = uf.find(t["source"])
        groups[t["group"]][t["cls"]] += 1
    gids = sorted(groups)
    gname = {g: f"g{i:03d}" for i, g in enumerate(gids)}
    print(f"sources: {len({t['source'] for t in tiles})}, copy links: {n_copy_links}, groups: {len(gids)}")
    sizes = sorted((sum(c.values()) for c in groups.values()), reverse=True)
    print(f"largest groups (tiles): {sizes[:5]}")

    # ---- greedy, count-based assignment
    rng = random.Random(SEED)
    totals = {c: sum(groups[g][c] for g in gids) for c in CLASSES}
    current = {s: Counter() for s in SPLITS}
    assign = {}

    def pick(cls):
        # split with the largest relative deficit for this class
        return max(SPLITS, key=lambda s: (FRACTIONS[s] * totals[cls] - current[s][cls]) / (FRACTIONS[s] * totals[cls]))

    healthy_groups = [g for g in gids if groups[g]["Healthy"] > 0]
    other_groups = [g for g in gids if groups[g]["Healthy"] == 0]
    for lst, cls in ((healthy_groups, "Healthy"), (other_groups, "Diseased")):
        rng.shuffle(lst)
        lst.sort(key=lambda g: -groups[g][cls])  # stable sort keeps the seeded order for ties
        for g in lst:
            s = pick(cls)
            assign[g] = s
            current[s] += groups[g]

    # ---- audit
    for t in tiles:
        t["split"] = assign[t["group"]]
    split_of_source = defaultdict(set)
    for t in tiles:
        split_of_source[t["source"]].add(t["split"])
    assert all(len(v) == 1 for v in split_of_source.values()), "a source spans splits"
    sha_split = defaultdict(set)
    for t in tiles:
        sha_split[t["blob_sha"]].add(t["split"])
    assert all(len(v) == 1 for v in sha_split.values()), "identical file spans splits"
    splits_arr = np.array([t["split"] for t in tiles])
    cross = 0
    examples = []
    for i, j in np.argwhere(np.triu(S, 1) >= NEARDUP_AUDIT_COS):
        if splits_arr[i] != splits_arr[j]:
            cross += 1
            if len(examples) < 10:
                examples.append((tiles[i]["file"], tiles[j]["file"], round(float(S[i, j]), 4)))

    summary = dict(
        upstream_repo="AnasBerka/CactiViT-materials",
        upstream_commit=upstream_commit,
        rules=dict(burst_seconds=BURST_SECONDS, copy_cos=COPY_COS, copy_scales=[1, 2, 4],
                   fractions=FRACTIONS, seed=SEED, assignment="greedy by relative deficit, Healthy groups first"),
        n_tiles=len(tiles), n_sources=len(split_of_source), n_groups=len(gids),
        per_split={s: dict(tiles=int(sum(current[s].values())),
                           Healthy=int(current[s]["Healthy"]), Diseased=int(current[s]["Diseased"]),
                           groups=sum(1 for g in gids if assign[g] == s),
                           healthy_groups=sum(1 for g in healthy_groups if assign[g] == s),
                           healthy_sources=len({t["source"] for t in tiles if t["split"] == s and t["cls"] == "Healthy"}),
                           upstream_folders=dict(Counter(t["upstream_folder"] for t in tiles if t["split"] == s)))
                   for s in SPLITS},
        audit=dict(sources_spanning_splits=0, identical_files_spanning_splits=0,
                   cross_split_tile_pairs_cos_ge_0_95=cross, cross_split_examples=examples),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write("file,cls,class_id,upstream_folder,blob_sha,source,group,split,old_split\n")
        for t in sorted(tiles, key=lambda t: (t["split"], t["cls"], t["file"])):
            f.write(",".join([t["file"], t["cls"], str(CLASSES.index(t["cls"])), t["upstream_folder"], t["blob_sha"],
                              t["source"], gname[t["group"]], t["split"], t["old_split"]]) + "\n")
    out.with_name(out.stem + "_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
