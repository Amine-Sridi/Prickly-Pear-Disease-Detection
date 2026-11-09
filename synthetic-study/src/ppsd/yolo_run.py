"""Resumable YOLO training + evaluation. Writes exactly one JSON per run.

Behaviour on re-run (e.g. after a Colab disconnect):
  JSON exists                    -> skip, return it
  train_done.json exists         -> skip training, (re)do evaluation
  weights/last.pt exists         -> resume training from it
  otherwise                      -> fresh training from yolov8n.pt

The model that is evaluated and reported is ALWAYS weights/last.pt (fixed epoch
count, no epoch selection on any split). Ultralytics still writes best.pt; it is
ignored on purpose.

CLI (used by the resume test in 00_setup):
  python -m ppsd.yolo_run --phase smoke --name x --imgsz 256 --seed 0 --epochs 3 \
      --data /content/datasets/yolo_split_v1/data.yaml --split-csv .../split_v1.csv --kind smoke
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

from . import configs
from .env import collect_env, seed_everything
from .metrics import NONE, tile_confusion
from .paths import Paths, get_paths
from .record import read_json, sha256_file, utc_now, write_json_atomic

SCHEMA = "ppsd-yolo-run-v1"


def _epochs_done(run_dir: Path) -> int:
    """Distinct epoch numbers in results.csv (a kill between the CSV write and the
    checkpoint write can make a resumed run log the same epoch twice)."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return 0
    with csv_path.open(newline="") as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    return len({r.get("epoch") for r in rows})


def _last_csv_row(run_dir: Path) -> dict:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="") as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    return rows[-1] if rows else {}


def _train(run_dir: Path, args: dict, data_yaml: str) -> dict:
    """Train or resume. Returns this attempt's log entry."""
    from ultralytics import YOLO

    last = run_dir / "weights" / "last.pt"
    attempt = {"started_utc": utc_now(), "epochs_done_before": _epochs_done(run_dir)}
    t0 = time.time()
    if last.exists():
        attempt["mode"] = "resume"
        try:
            # explicit path: a bare resume=True can fall back to "latest run anywhere"
            YOLO(str(last)).train(resume=str(last), data=data_yaml, workers=args.get("workers", 2))
        except AssertionError as e:  # all epochs were already done before the disconnect
            if "nothing to resume" not in str(e):
                raise
            attempt["note"] = "checkpoint already complete; nothing to resume"
    else:
        attempt["mode"] = "fresh"
        YOLO(configs.BASE_MODEL).train(data=data_yaml, project=str(run_dir.parent), name=run_dir.name, **args)
    attempt["seconds"] = round(time.time() - t0, 1)
    attempt["finished_utc"] = utc_now()
    attempt["epochs_done_after"] = _epochs_done(run_dir)
    return attempt


def _ultralytics_eval(weights: Path, data_yaml: str, split: str, imgsz: int, out_dir: Path) -> dict:
    from ultralytics import YOLO

    m = YOLO(str(weights))
    met = m.val(data=data_yaml, split=split, imgsz=imgsz, project=str(out_dir), name=f"eval_{split}",
                exist_ok=True, verbose=False, **configs.EVAL_ARGS)
    names = met.names
    per_class = {}
    for i, c in enumerate(met.box.ap_class_index):
        p, r, ap50, ap = met.box.class_result(i)
        per_class[names[int(c)]] = {"precision": float(p), "recall": float(r),
                                    "map50": float(ap50), "map50_95": float(ap)}
    mp, mr, map50, map_ = (float(x) for x in met.box.mean_results())
    return {
        "per_class": per_class,
        "all": {"precision": mp, "recall": mr, "map50": map50, "map50_95": map_},
        "note": "Ultralytics P/R are at the single confidence that maximises mean F1 over classes; AP is threshold-free",
        "eval_args": {**configs.EVAL_ARGS, "imgsz": imgsz, "split": split},
    }


def _tile_eval(weights: Path, dataset_dir: Path, split: str, split_rows: list[dict],
               imgsz: int) -> tuple[dict, dict]:
    from ultralytics import YOLO

    m = YOLO(str(weights))
    rows = [r for r in split_rows if r["split"] == split]
    paths = [str(dataset_dir / "images" / split / r["file"]) for r in rows]
    truth = {r["file"]: r["cls"] for r in rows}
    preds = {}
    for res in m.predict(source=paths, imgsz=imgsz, conf=0.001, iou=0.7, augment=False, stream=True,
                         verbose=False, batch=32):
        name = Path(res.path).name
        if res.boxes is None or len(res.boxes) == 0:
            preds[name] = NONE
        else:
            preds[name] = res.names[int(res.boxes.cls[int(res.boxes.conf.argmax())])]
    if set(preds) != set(truth):
        raise RuntimeError(f"tile eval covered {len(preds)} of {len(truth)} images")
    files = sorted(truth)
    out = tile_confusion([truth[f] for f in files], [preds[f] for f in files])
    out["predictions_csv"] = f"tile_predictions_{split}.csv"
    return out, {f: preds[f] for f in files}


def run_yolo(*, phase: str, name: str, data_yaml: str, split_csv: str, imgsz: int, seed: int,
             kind: str = "experiment", notes: str = "", paths: Paths | None = None,
             train_overrides: dict | None = None, extra_record: dict | None = None,
             eval_splits=("val", "test")) -> dict:
    from .dataset import read_split_csv

    paths = (paths or get_paths()).make()
    run_dir = paths.yolo_runs / phase / name
    json_path = paths.results / phase / f"{name}.json"
    if json_path.exists():
        print(f"[skip] {json_path} already exists")
        return read_json(json_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    args = configs.train_args(imgsz=imgsz, seed=seed, **(train_overrides or {}))
    unknown = configs.check_args_known(args)
    if unknown:
        raise ValueError(f"installed Ultralytics does not know these args: {unknown}")

    attempts_path = run_dir / "attempts.json"
    attempts = read_json(attempts_path) if attempts_path.exists() else []
    done_marker = run_dir / "train_done.json"
    seed_everything(seed)
    if not done_marker.exists():
        attempts.append({"started_utc": utc_now(), "mode": "unknown", "status": "running"})
        write_json_atomic(attempts_path, attempts)  # so an interrupted attempt is visible later
        entry = _train(run_dir, args, data_yaml)
        attempts[-1] = {**entry, "status": "finished"}
        write_json_atomic(attempts_path, attempts)
        n = _epochs_done(run_dir)
        if n < args["epochs"]:
            raise RuntimeError(f"training stopped after {n}/{args['epochs']} epochs; re-run to resume")
        write_json_atomic(done_marker, {"finished_utc": utc_now(), "epochs": n})

    weights = run_dir / "weights" / "last.pt"
    dataset_dir = Path(data_yaml).parent
    rows = read_split_csv(split_csv)
    evaluation = {}
    t_eval = time.time()
    for split in eval_splits:
        ul = _ultralytics_eval(weights, data_yaml, split, imgsz, run_dir)
        tile, preds = _tile_eval(weights, dataset_dir, split, rows, imgsz)
        with (run_dir / tile["predictions_csv"]).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "true", "pred"])
            truth = {r["file"]: r["cls"] for r in rows if r["split"] == split}
            for fname, p in preds.items():
                w.writerow([fname, truth[fname], p])
        evaluation[split] = {"ultralytics": ul, "tile_level": tile}

    finished_attempts = [a for a in attempts if a.get("status") == "finished"]
    record = {
        "schema": SCHEMA,
        "kind": kind,  # "smoke" records are tests of the pipeline, never results
        "phase": phase,
        "run_name": name,
        "notes": notes,
        "created_utc": utc_now(),
        "seed": seed,
        "config": {"base_model": configs.BASE_MODEL, "train_args": args,
                   "evaluated_weights": "last.pt (fixed epoch count, no selection)"},
        "data": {
            "data_yaml": data_yaml,
            "split_csv": Path(split_csv).name,
            "split_csv_sha256": sha256_file(split_csv),
            "dataset_manifest": read_json(dataset_dir / "dataset_manifest.json")
            if (dataset_dir / "dataset_manifest.json").exists() else None,
        },
        "training": {
            "epochs_completed": _epochs_done(run_dir),
            "attempts": attempts,
            "n_resumes": sum(1 for a in finished_attempts if a.get("mode") == "resume"),
            "interrupted_attempts": sum(1 for a in attempts if a.get("status") != "finished"),
            "train_seconds_recorded": round(sum(a.get("seconds", 0) for a in finished_attempts), 1),
            "last_epoch_row": _last_csv_row(run_dir),
            "weights_sha256": sha256_file(weights),
        },
        "eval": evaluation,
        "eval_seconds": round(time.time() - t_eval, 1),
        "env": collect_env(),
        "run_dir": str(run_dir),
    }
    if extra_record:
        record.update(extra_record)
    write_json_atomic(json_path, record)
    write_json_atomic(run_dir / "run_record.json", record)
    print(f"[done] {json_path}")
    return record


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--kind", default="experiment")
    ap.add_argument("--notes", default="")
    ap.add_argument("--eval-splits", default="val,test", help="comma list; smoke runs use 'val' only")
    a = ap.parse_args()
    over = {"epochs": a.epochs} if a.epochs else {}
    run_yolo(phase=a.phase, name=a.name, data_yaml=a.data, split_csv=a.split_csv, imgsz=a.imgsz, seed=a.seed,
             kind=a.kind, notes=a.notes, train_overrides=over, eval_splits=tuple(a.eval_splits.split(",")))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    _cli()
