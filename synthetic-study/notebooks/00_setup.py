# %% [markdown]
# # 00_setup: Colab environment, Drive layout, dataset, smoke test
#
# Run top to bottom on a **T4 GPU runtime** (Runtime > Change runtime type > T4 GPU).
# Every cell can be re-run after a disconnect; finished work is detected and skipped.
#
# What this notebook does
# 1. checks the GPU (does not stop if there is none; the dataset steps still work)
# 2. mounts Google Drive and creates the fixed folder layout
# 3. gets the study code (`synthetic-study/`)
# 4. installs the pinned libraries and checks the versions
# 5. writes an environment snapshot JSON
# 6. builds (first time) or restores (later) the grouped-split dataset and verifies every file hash
# 7. **smoke test** (pipeline check, *not a result*): 1 epoch at 256 px and 1 epoch at 640 px
# 8. **resume test**: starts a 3-epoch run, kills it after epoch 1 (simulated disconnect), re-runs it,
#    and checks that it resumed instead of restarting
#
# What to send back to Claude (all on Drive under `prickly-pear-sd/outputs/`):
# - `env/env_*.json`
# - `runs/smoke/*.json` (3 files)
# - the text printed by the last cell

# %% 1. Settings
ROOT = "/content/drive/MyDrive/prickly-pear-sd"  # everything persistent lives here
CODE_SOURCE = "github"  # "github" (clone branch below) or "drive" (folder uploaded to ROOT/code/synthetic-study)
REPO_URL = "https://github.com/Amine-Sridi/Prickly-Pear-Disease-Detection.git"
BRANCH = "synthetic-study"
SPLIT_NAME = "split_v1"
RUN_SMOKE = True
RUN_RESUME_TEST = True

# %% 2. GPU check (informative only)
import subprocess

smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(smi.stdout if smi.returncode == 0 else "No GPU visible. Setup and dataset steps still work; skip 7-8.")
try:
    import torch
    print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False

# %% 3. Mount Google Drive and create the folder layout
import os
from google.colab import drive

drive.mount("/content/drive")
os.environ["PPSD_ROOT"] = ROOT
os.makedirs(ROOT, exist_ok=True)

# %% 4. Get the study code
import shutil
import sys
from pathlib import Path

CODE_DIR = Path("/content/ppsd_code")
if CODE_SOURCE == "github":
    if not (CODE_DIR / ".git").exists():
        # partial + sparse clone: only synthetic-study/, not the 700 MB of old checkpoints
        subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", "--branch", BRANCH, REPO_URL,
                        str(CODE_DIR)], check=True)
        subprocess.run(["git", "-C", str(CODE_DIR), "sparse-checkout", "set", "synthetic-study"], check=True)
    else:
        subprocess.run(["git", "-C", str(CODE_DIR), "pull", "--ff-only"], check=True)
    CODE_COMMIT = subprocess.run(["git", "-C", str(CODE_DIR), "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
else:
    src = Path(ROOT) / "code" / "synthetic-study"
    assert src.exists(), f"upload the synthetic-study folder to {src}"
    if CODE_DIR.exists():
        shutil.rmtree(CODE_DIR)
    shutil.copytree(src, CODE_DIR / "synthetic-study")
    CODE_COMMIT = "uploaded-from-drive"

STUDY = CODE_DIR / "synthetic-study"
SRC = str(STUDY / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
SPLIT_CSV = str(STUDY / "splits" / f"{SPLIT_NAME}.csv")
print("code commit:", CODE_COMMIT)
print("split CSV:", SPLIT_CSV)

# %% 5. Install pinned libraries (torch is kept as provided by Colab)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(STUDY / "requirements-colab.txt")],
               check=True)
from ppsd.env import check_pins

bad = check_pins()
if bad:
    print("Version mismatch (wanted, installed):", bad)
    print("=> Runtime > Restart session, then run again from cell 1 (installed packages persist).")
else:
    print("All pinned versions installed.")

# %% 6. Folder layout, environment snapshot, seed helper
from ppsd.env import collect_env, pip_freeze, seed_everything
from ppsd.paths import get_paths
from ppsd.record import utc_now, write_json_atomic

P = get_paths().make()
for p in (P.data, P.yolo_runs, P.lora, P.synthetic, P.results, P.env, P.logs):
    print("ok", p)

stamp = utc_now().replace(":", "").replace("-", "")
env = collect_env()
env["code_commit"] = CODE_COMMIT
write_json_atomic(P.env / f"env_{stamp}.json", env)
(P.env / f"pip_freeze_{stamp}.txt").write_text(pip_freeze())
print({k: env["packages"][k] for k in ("torch", "ultralytics", "diffusers", "transformers", "accelerate", "peft")})
print("GPU:", env["gpu"].get("name"), env["gpu"].get("total_memory_gb"), "GB")

seed_everything(0)
import random
import numpy as np

print("seed check (must print the same numbers every time):", random.random(), np.random.rand())

# %% 7. Dataset: restore from Drive, or build once from the public CactiViT repository
from ppsd.dataset import ensure_dataset

LOCAL_DS = P.local_data / f"yolo_{SPLIT_NAME}"
DRIVE_ZIP = P.data / f"yolo_{SPLIT_NAME}.zip"
info = ensure_dataset(SPLIT_CSV, LOCAL_DS, DRIVE_ZIP)
DATA_YAML = str(LOCAL_DS / "data.yaml")
print(info)
print(open(DATA_YAML).read())

# %% 8. Smoke test: 1 epoch at 256 px and 640 px (pipeline check + timing; NOT a result)
from ppsd.yolo_run import run_yolo

os.chdir("/content")  # Ultralytics downloads yolov8n.pt into the working directory
smoke = {}
if RUN_SMOKE and HAS_GPU:
    for imgsz in (256, 640):
        smoke[imgsz] = run_yolo(
            phase="smoke", name=f"smoke_1ep_img{imgsz}_s0", data_yaml=DATA_YAML, split_csv=SPLIT_CSV,
            imgsz=imgsz, seed=0, kind="smoke", notes="1-epoch pipeline/timing check, not a result",
            train_overrides={"epochs": 1}, eval_splits=("val",))
        t = smoke[imgsz]["training"]
        print(imgsz, "px: train", t["train_seconds_recorded"], "s for", t["epochs_completed"], "epoch;",
              "eval", smoke[imgsz]["eval_seconds"], "s")
else:
    print("skipped (no GPU or RUN_SMOKE=False)")

# %% 9. Resume test: kill a 3-epoch run after epoch 1, re-run, check that it resumed
import csv as _csv
import signal
import time

from ppsd.record import read_json

name = "resume_test_3ep_img256_s0"
cmd = [sys.executable, "-m", "ppsd.yolo_run", "--phase", "smoke", "--name", name, "--data", DATA_YAML,
       "--split-csv", SPLIT_CSV, "--imgsz", "256", "--seed", "0", "--epochs", "3", "--kind", "smoke",
       "--eval-splits", "val", "--notes", "resume test, not a result"]
child_env = {**os.environ, "PYTHONPATH": SRC, "PPSD_ROOT": ROOT}
run_dir = P.yolo_runs / "smoke" / name
json_path = P.results / "smoke" / f"{name}.json"


def epochs_logged():
    f = run_dir / "results.csv"
    if not f.exists():
        return 0
    with f.open() as fh:
        return len({r.get("epoch", r.get("                  epoch")) for r in _csv.DictReader(fh)})


if RUN_RESUME_TEST and HAS_GPU and not json_path.exists():
    if not (run_dir / "weights" / "last.pt").exists():
        proc = subprocess.Popen(cmd, env=child_env, cwd="/content", start_new_session=True)
        while proc.poll() is None:
            if epochs_logged() >= 1 and (run_dir / "weights" / "last.pt").exists():
                time.sleep(10)  # let the checkpoint write finish
                os.killpg(proc.pid, signal.SIGKILL)  # simulated disconnect
                print("killed after", epochs_logged(), "epoch(s)")
                break
            time.sleep(3)
        proc.wait()
    subprocess.run(cmd, env=child_env, cwd="/content", check=True)  # must resume, not restart

if json_path.exists():
    rec = read_json(json_path)
    tr = rec["training"]
    print("epochs:", tr["epochs_completed"], "| resumes:", tr["n_resumes"],
          "| interrupted attempts:", tr["interrupted_attempts"])
    assert tr["epochs_completed"] == 3, "resume test failed: wrong epoch count"
    assert tr["n_resumes"] >= 1, "resume test failed: the second call did not resume"
    print("RESUME TEST PASSED")

# %% 10. Summary to send back
import glob
import json

print("ENV FILES:", sorted(glob.glob(str(P.env / "env_*.json")))[-1:])
for f in sorted(glob.glob(str(P.results / "smoke" / "*.json"))):
    r = json.load(open(f))
    t = r["training"]
    v = r["eval"]["val"]
    print(Path(f).name, "| imgsz", r["config"]["train_args"]["imgsz"], "| epochs", t["epochs_completed"],
          "| train s", t["train_seconds_recorded"], "| eval s", r["eval_seconds"],
          "| GPU", r["env"]["gpu"].get("name"),
          "| tile confusion (val)", v["tile_level"]["confusion"])
print("\nPlease send: the env JSON above, the files in", P.results / "smoke", "and this printout.")
