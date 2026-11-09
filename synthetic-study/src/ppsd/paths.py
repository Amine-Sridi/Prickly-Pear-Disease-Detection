"""Fixed folder layout on Google Drive (persistent) and on the Colab VM (fast, wiped).

Drive root (override with env var PPSD_ROOT):
  data/                    dataset zips (built once, unzipped to the VM each session)
  checkpoints/yolo/<phase>/<run>/   Ultralytics run folders (weights/last.pt = resume point)
  checkpoints/lora/        LoRA checkpoints (Phase 3)
  synthetic/               generated images + per-image settings CSV (Phase 3)
  outputs/runs/<phase>/    ONE JSON per run (the only source of reported numbers)
  outputs/env/             environment snapshots
  logs/                    free-form logs
VM:
  /content/datasets/<name>/   unzipped dataset used for training (override with PPSD_LOCAL_DATA)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = "/content/drive/MyDrive/prickly-pear-sd"
DEFAULT_LOCAL_DATA = "/content/datasets"


@dataclass(frozen=True)
class Paths:
    root: Path
    local_data: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def yolo_runs(self) -> Path:
        return self.checkpoints / "yolo"

    @property
    def lora(self) -> Path:
        return self.checkpoints / "lora"

    @property
    def synthetic(self) -> Path:
        return self.root / "synthetic"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def results(self) -> Path:
        return self.outputs / "runs"

    @property
    def env(self) -> Path:
        return self.outputs / "env"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def make(self) -> "Paths":
        for p in (self.data, self.yolo_runs, self.lora, self.synthetic, self.results, self.env, self.logs,
                  self.local_data):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(root: str | os.PathLike | None = None, local_data: str | os.PathLike | None = None) -> Paths:
    return Paths(
        root=Path(root or os.environ.get("PPSD_ROOT", DEFAULT_ROOT)),
        local_data=Path(local_data or os.environ.get("PPSD_LOCAL_DATA", DEFAULT_LOCAL_DATA)),
    )
