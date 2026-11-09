"""ppsd: helpers for the prickly-pear Stable Diffusion augmentation study.

Modules
  paths      Google Drive folder layout
  env        seed helper, environment / GPU capture
  record     atomic JSON writing, hashing, timestamps
  dataset    build and verify the YOLO dataset from split CSV + upstream repo
  metrics    tile-level confusion / per-class precision and recall
  configs    the original training recipe (from train3/args.yaml)
  yolo_run   resumable YOLO training + evaluation -> one JSON per run
"""

__version__ = "0.1.0"
