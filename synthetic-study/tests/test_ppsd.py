"""CPU-only tests (no Ultralytics / GPU needed). Run from the repository root:
    python synthetic-study/tests/test_ppsd.py          (plain asserts)
or  python -m pytest synthetic-study/tests -q          (if pytest is installed)
"""
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "synthetic-study"
sys.path.insert(0, str(STUDY / "src"))
sys.path.insert(0, str(STUDY / "tools"))

from ppsd import configs, dataset, metrics, record  # noqa: E402
from ppsd.yolo_run import _epochs_done  # noqa: E402

SPLIT_CSV = STUDY / "splits" / "split_v1.csv"
LOCAL_IMAGES = REPO / "DatasetPreped2" / "images"


def test_git_blob_sha_matches_git():
    f = STUDY / "requirements-colab.txt"
    want = subprocess.run(["git", "hash-object", str(f)], capture_output=True, text=True, check=True).stdout.strip()
    assert record.git_blob_sha(f.read_bytes()) == want


def test_tile_confusion():
    y_true = ["Healthy", "Healthy", "Healthy", "Diseased", "Diseased"]
    y_pred = ["Healthy", "Diseased", "none", "Diseased", "Diseased"]
    m = metrics.tile_confusion(y_true, y_pred)
    assert m["confusion"]["true_Healthy"] == {"pred_Healthy": 1, "pred_Diseased": 1, "pred_none": 1}
    assert abs(m["per_class"]["Healthy"]["recall"] - 1 / 3) < 1e-9
    assert m["per_class"]["Healthy"]["precision"] == 1.0
    assert abs(m["per_class"]["Diseased"]["precision"] - 2 / 3) < 1e-9
    assert abs(m["balanced_accuracy"] - (1 / 3 + 1) / 2) < 1e-9
    never = metrics.tile_confusion(["Healthy", "Diseased"], ["Diseased", "Diseased"])
    assert never["per_class"]["Healthy"]["precision"] is None  # never predicted: undefined, not 0 or 1
    try:
        metrics.tile_confusion(["Healthy"], ["cat"])
        raise AssertionError("unknown label accepted")
    except ValueError:
        pass


def test_split_integrity():
    rows = dataset.read_split_csv(SPLIT_CSV)
    assert len(rows) == 3252
    for key in ("source", "group", "blob_sha"):
        spans = defaultdict(set)
        for r in rows:
            spans[r[key]].add(r["split"])
        assert all(len(v) == 1 for v in spans.values()), f"{key} spans several splits"
    summary = json.loads((STUDY / "splits" / "split_v1_summary.json").read_text())
    for s in ("train", "val", "test"):
        for c in ("Healthy", "Diseased"):
            assert sum(r["split"] == s and r["cls"] == c for r in rows) == summary["per_split"][s][c]
    assert summary["audit"]["cross_split_tile_pairs_cos_ge_0_95"] == 0


def test_build_verify_zip_roundtrip():
    rows = dataset.read_split_csv(SPLIT_CSV)
    pick = []
    for s in ("train", "val", "test"):
        for c in ("Healthy", "Diseased"):
            pick += [r for r in rows if r["split"] == s and r["cls"] == c][:2]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        up = td / "upstream"
        for r in pick:  # fake upstream layout from the local copy of the same files
            src = next(LOCAL_IMAGES.glob(f"*/*/{r['file']}"))
            dst = up / dataset.UPSTREAM_SUBDIR / r["upstream_folder"] / r["file"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        sub_csv = td / "sub.csv"
        with open(SPLIT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            keep = {r["file"] for r in pick}
            body = [row for row in reader if row[0] in keep]
        with open(sub_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(header)
            w.writerows(body)
        out = td / "ds"
        m = dataset.build_yolo_dataset(sub_csv, up, out)
        assert m["counts"]["val"] == {"Healthy": 2, "Diseased": 2}
        assert dataset.verify_yolo_dataset(out, sub_csv)["images"] == len(pick)
        lbl = next((out / "labels" / "train").glob("*.txt")).read_text().strip()
        assert lbl.endswith("0.5 0.5 1.0 1.0") and lbl.split()[0] in ("0", "1")
        y = (out / "data.yaml").read_text()
        assert "train: images/train" in y and "0: Healthy" in y and "1: Diseased" in y
        # zip -> unzip gives a dataset that still verifies
        z = dataset.zip_dataset(out, td / "ds.zip")
        restored = dataset.unzip_dataset(z, td / "restored")
        assert dataset.verify_yolo_dataset(restored, sub_csv)["ok"]
        # tampering is detected
        victim = next((restored / "images" / "test").glob("*.png"))
        victim.write_bytes(victim.read_bytes() + b"x")
        try:
            dataset.verify_yolo_dataset(restored, sub_csv)
            raise AssertionError("tampered image not detected")
        except ValueError:
            pass
        # a corrupted upstream file is refused at build time
        bad = next((up / dataset.UPSTREAM_SUBDIR).rglob("*.png"))
        bad.write_bytes(b"not a png")
        try:
            dataset.build_yolo_dataset(sub_csv, up, td / "ds2")
            raise AssertionError("corrupted upstream file accepted")
        except ValueError:
            pass


def test_epochs_done_counts_distinct_epochs():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        assert _epochs_done(d) == 0
        (d / "results.csv").write_text("epoch,time,metrics/mAP50(B)\n1,10,0.5\n2,20,0.5\n2,25,0.5\n")
        assert _epochs_done(d) == 2


def test_train_args():
    a = configs.train_args(imgsz=256, seed=3)
    assert a["imgsz"] == 256 and a["seed"] == 3 and a["optimizer"] == "AdamW" and a["lr0"] == 0.001667
    assert a["augment"] is False and a["epochs"] == 20 and a["batch"] == 16
    assert configs.train_args(640, 0, epochs=1)["epochs"] == 1


def test_write_json_atomic():
    with tempfile.TemporaryDirectory() as td:
        p = record.write_json_atomic(Path(td) / "a" / "b.json", {"x": {1, 2}, "p": Path("q")})
        assert json.loads(p.read_text()) == {"x": [1, 2], "p": "q"}
        assert not list(Path(td).rglob("*.tmp"))


def test_notebook_generation():
    import py2nb

    nb_path = py2nb.convert(STUDY / "notebooks" / "00_setup.py")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert nb["cells"][0]["cell_type"] == "markdown" and len(code) >= 9
    for c in code:
        compile("".join(c["source"]), "<cell>", "exec")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
