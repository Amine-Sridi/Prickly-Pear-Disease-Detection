"""Build / verify the YOLO dataset for a frozen split CSV.

Source of images: the public CactiViT-materials repository (MIT), pinned to one
commit. Every file is checked against the git blob SHA stored in the split CSV,
so the dataset on Colab is byte-identical to the one audited in Phase 0.

Layout produced:
  <out>/images/{train,val,test}/<file>.png
  <out>/labels/{train,val,test}/<stem>.txt     "<class_id> 0.5 0.5 1.0 1.0"  (whole-tile box, as in the original)
  <out>/data.yaml                               absolute paths, names {0: Healthy, 1: Diseased}
  <out>/dataset_manifest.json                   counts, split CSV hash, upstream commit
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

from .record import git_blob_sha, read_json, sha256_file, utc_now, write_json_atomic

UPSTREAM_URL = "https://github.com/AnasBerka/CactiViT-materials.git"
UPSTREAM_COMMIT = "730447f9bc1b05bbd58377c4fa271bc20583963e"
UPSTREAM_SUBDIR = "Datasets/DBX"
USED_FOLDERS = ("Healthy", "Damaged", "EarlyStage", "LateStage", "Old_Dead")
NAMES = {0: "Healthy", 1: "Diseased"}
SPLITS = ("train", "val", "test")
WHOLE_TILE_BOX = "0.5 0.5 1.0 1.0"


def read_split_csv(path: str | os.PathLike) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["class_id"] = int(r["class_id"])
        if r["split"] not in SPLITS:
            raise ValueError(f"bad split {r['split']!r} for {r['file']}")
        if NAMES[r["class_id"]] != r["cls"]:
            raise ValueError(f"class id / name mismatch for {r['file']}")
    names = [r["file"] for r in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate file names in split CSV")
    return rows


def fetch_upstream(dest: str | os.PathLike, commit: str = UPSTREAM_COMMIT) -> Path:
    """Partial, sparse clone of only the 5 folders used (skips Confused/NoCactus). Needs git + network."""
    dest = Path(dest)
    if (dest / ".git").exists():
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head == commit:
            return dest
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def git(*a, cwd=None):
        subprocess.run(["git", *a], cwd=cwd, check=True)

    git("clone", "--filter=blob:none", "--no-checkout", "--quiet", UPSTREAM_URL, str(dest))
    git("sparse-checkout", "init", "--cone", cwd=dest)
    git("sparse-checkout", "set", *[f"{UPSTREAM_SUBDIR}/{f}" for f in USED_FOLDERS], cwd=dest)
    git("checkout", "--quiet", commit, cwd=dest)
    return dest


def write_data_yaml(dataset_dir: str | os.PathLike, train: str = "images/train",
                    yaml_name: str = "data.yaml") -> Path:
    """`train` may be a folder or a .txt list of image paths (used later for oversampling / synthetic)."""
    d = Path(dataset_dir).resolve()
    text = (
        f"path: {d.as_posix()}\n"
        f"train: {train}\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n" + "".join(f"  {k}: {v}\n" for k, v in NAMES.items())
    )
    p = d / yaml_name
    p.write_text(text, encoding="utf-8")
    return p


def build_yolo_dataset(split_csv: str | os.PathLike, upstream_root: str | os.PathLike,
                       out_dir: str | os.PathLike) -> dict:
    rows = read_split_csv(split_csv)
    src_root = Path(upstream_root) / UPSTREAM_SUBDIR
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    for s in SPLITS:
        (tmp / "images" / s).mkdir(parents=True)
        (tmp / "labels" / s).mkdir(parents=True)
    for r in rows:
        src = src_root / r["upstream_folder"] / r["file"]
        data = src.read_bytes()
        if git_blob_sha(data) != r["blob_sha"]:
            raise ValueError(f"hash mismatch for {src}")
        (tmp / "images" / r["split"] / r["file"]).write_bytes(data)
        (tmp / "labels" / r["split"] / (Path(r["file"]).stem + ".txt")).write_text(
            f"{r['class_id']} {WHOLE_TILE_BOX}\n", encoding="utf-8")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)
    write_data_yaml(out_dir)
    manifest = {
        "built_utc": utc_now(),
        "split_csv": Path(split_csv).name,
        "split_csv_sha256": sha256_file(split_csv),
        "upstream_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "counts": {s: dict(Counter(r["cls"] for r in rows if r["split"] == s)) for s in SPLITS},
        "label_rule": f"one whole-tile box per image: '<class_id> {WHOLE_TILE_BOX}'",
    }
    write_json_atomic(out_dir / "dataset_manifest.json", manifest)
    return manifest


def verify_yolo_dataset(dataset_dir: str | os.PathLike, split_csv: str | os.PathLike) -> dict:
    """Re-hash every image and re-read every label. Raises on any mismatch."""
    d = Path(dataset_dir)
    rows = read_split_csv(split_csv)
    expected = {(r["split"], r["file"]): r for r in rows}
    found = {(s, p.name) for s in SPLITS for p in (d / "images" / s).glob("*.png")}
    missing, extra = set(expected) - found, found - set(expected)
    if missing or extra:
        raise ValueError(f"dataset mismatch: {len(missing)} missing, {len(extra)} unexpected images")
    for (s, f), r in expected.items():
        if git_blob_sha((d / "images" / s / f).read_bytes()) != r["blob_sha"]:
            raise ValueError(f"hash mismatch: {s}/{f}")
        label = (d / "labels" / s / (Path(f).stem + ".txt")).read_text(encoding="utf-8").strip()
        if label != f"{r['class_id']} {WHOLE_TILE_BOX}":
            raise ValueError(f"bad label for {s}/{f}: {label!r}")
    m = read_json(d / "dataset_manifest.json")
    if m["split_csv_sha256"] != sha256_file(split_csv):
        raise ValueError("dataset was built from a different split CSV")
    return {"ok": True, "images": len(found), "counts": m["counts"]}


def zip_dataset(dataset_dir: str | os.PathLike, zip_path: str | os.PathLike) -> Path:
    """PNG does not compress further, so store uncompressed (fast)."""
    d, zip_path = Path(dataset_dir), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.name != "data.yaml":  # data.yaml holds absolute paths; rewritten on unzip
                z.write(p, p.relative_to(d).as_posix())
    os.replace(tmp, zip_path)
    return zip_path


def unzip_dataset(zip_path: str | os.PathLike, dataset_dir: str | os.PathLike) -> Path:
    d = Path(dataset_dir)
    tmp = d.with_name(d.name + ".unzipping")
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    if d.exists():
        shutil.rmtree(d)
    os.replace(tmp, d)
    write_data_yaml(d)
    return d


def ensure_dataset(split_csv: str | os.PathLike, local_dir: str | os.PathLike, drive_zip: str | os.PathLike,
                   upstream_cache: str | os.PathLike = "/content/upstream_cactivit") -> dict:
    """Fastest valid path: existing VM copy -> unzip from Drive -> build from upstream (then zip to Drive)."""
    local_dir, drive_zip = Path(local_dir), Path(drive_zip)
    if local_dir.exists():
        try:
            return {"source": "vm", **verify_yolo_dataset(local_dir, split_csv)}
        except Exception as e:  # noqa: BLE001 - rebuild on any problem
            print(f"VM copy invalid ({e}); rebuilding")
    if drive_zip.exists():
        unzip_dataset(drive_zip, local_dir)
        return {"source": "drive_zip", **verify_yolo_dataset(local_dir, split_csv)}
    up = fetch_upstream(upstream_cache)
    build_yolo_dataset(split_csv, up, local_dir)
    res = verify_yolo_dataset(local_dir, split_csv)
    zip_dataset(local_dir, drive_zip)
    return {"source": "built_from_upstream", **res}
