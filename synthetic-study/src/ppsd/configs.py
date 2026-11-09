"""The original training recipe, made explicit.

Source: datasets/runs/detect/train3/args.yaml (Ultralytics 8.3.29) plus what
`optimizer=auto` resolved to in that version (ultralytics/engine/trainer.py
v8.3.29, lines 773-782, and the train4 log: "AdamW(lr=0.001667, momentum=0.9)"):
  AdamW, lr0 = round(0.002*5/(4+nc), 6) = 0.001667 for nc=2, momentum 0.9,
  warmup_bias_lr forced to 0.0.
Everything is passed explicitly so that default changes in newer Ultralytics
versions cannot silently change the recipe.

Deliberate differences from the original (none changes the training maths):
  augment=False   original passed augment=True, which in Ultralytics only turns on
                  test-time augmentation in the final validation (validator.py:
                  `augment = self.args.augment and (not self.training)`).
                  We evaluate without TTA.
  save_period=-1  original saved every epoch (save_period=1); last.pt is still written
                  every epoch, which is all resuming needs.
  workers=2       Colab has 2 vCPUs (original: 8).
"""
from __future__ import annotations

BASE_MODEL = "yolov8n.pt"

ORIGINAL_TRAIN_ARGS = {
    "epochs": 20,
    "batch": 16,
    "imgsz": 640,
    "patience": 100,
    "optimizer": "AdamW",
    "lr0": 0.001667,
    "lrf": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0,
    "cos_lr": False,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "nbs": 64,
    "close_mosaic": 10,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "cutmix": 0.0,
    "dropout": 0.0,
    "multi_scale": False,
    "rect": False,
    "single_cls": False,
    "fraction": 1.0,
    "pretrained": True,
    "amp": True,
    "deterministic": True,
    "freeze": None,
}

RUN_ARGS = {
    "augment": False,
    "save_period": -1,
    "workers": 2,
    "val": True,
    "plots": True,
    "exist_ok": True,
    "verbose": True,
}

EVAL_ARGS = {"batch": 16, "conf": 0.001, "iou": 0.7, "augment": False, "plots": True, "workers": 2}


def train_args(imgsz: int, seed: int, **overrides) -> dict:
    a = {**ORIGINAL_TRAIN_ARGS, **RUN_ARGS, "imgsz": imgsz, "seed": seed}
    a.update(overrides)
    return a


def check_args_known(args: dict) -> list[str]:
    """Keys that the installed Ultralytics does not know (call on Colab; empty list = OK)."""
    from ultralytics.cfg import DEFAULT_CFG_DICT
    return sorted(k for k in args if k not in DEFAULT_CFG_DICT)
