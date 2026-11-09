# RUNLOG

Every decision, with its reason. Newest at the bottom. Numbers here only come from runs or files that exist.

## 2026-11-07: Phase 0 (audit, no project code changed)

- **Study files go in a new folder `synthetic-study/`.** Reason: the brief says to keep new work clearly separated and not rewrite the original code.
- **Audit done locally, read-only, on the untracked `DatasetPreped2/`.** Scripts are in `phase0/`. Reason: light analysis is allowed on the laptop; no GPU was used.
- **Python flag:** the audit scripts were run with `python -P -E` instead of `-I`, because `-I` hides the user site-packages where numpy/PIL are installed. The scripts only read the repo's own image files.
- **Dataset identified as CactiViT-materials (MIT).** Reason: the notebook path `Datasets/DBX/Cactus`, the folder names, and an exact file-name match against the upstream tree (123/123 Healthy, 3129/3129 Diseased).
- **Finding: whole-tile boxes, photo-level leakage (~97%), 22 cross-split exact duplicates, Healthy = 123 tiles / 65 photos.** Details in `UNDERSTANDING.md`.
- **Finding: `train` ≡ `train3`, `train2` is a 3-epoch prefix, `val/` is the untrained COCO model.** So the existing evidence is 2 configurations × 1 seed.
- **Proposed (not adopted): evaluation option B** (grouped re-split, fixed 20 epochs with `last.pt`, frozen real-only test). Waiting for Amine.
- **Nothing pushed, nothing committed.**

### Amine's answers (2026-11-07)

1. Evaluation: **option B** adopted.
2. Image size: "don't know" → Claude's recommendation adopted (see below).
3. Competition was two years ago; the study is an independent improvement. No competition constraint.
4. Framing: "what do you think is best" → Claude's recommendation adopted (see below).
5. Drive space: not a concern.

## 2026-11-07: Phase 1 (Colab setup)

- **Split v1 built** (`scripts/make_grouped_split.py` → `splits/split_v1.csv`, `split_v1_summary.json`).
  Groups = source photo ∪ copies of the same photo ∪ capture bursts. Greedy count-based assignment, 60/20/20, seed 0; no model output used.
  Result: Healthy 73/25/25, Diseased 1815/708/606 (train/val/test); 175 groups; 0 sources, 0 identical files and 0 tile pairs with cosine ≥ 0.95 across splits.
- **Burst threshold 30 s, not 10 s.** With 10 s, one cross-split near-duplicate pair remained (`img_20210327_172718` / `_172741`, 23 s apart, cosine 0.95). With 30 s there are none, and Healthy counts stay 73/25/25. Cost: healthy groups in val/test drop from 9/9 to 7/8, and Diseased is less even. Removing leakage is the point of option B.
- **Copy detection:** 11 source pairs are the same photo (8 WhatsApp↔UUID, 3 UUID↔timestamped at ¼ scale). All matches have coordinate-consistent tiles (scale 1/2/4); no sky-tile false matches at cosine 0.97.
- **Dataset source on Colab = public CactiViT-materials at commit 730447f.** All 3252 local tiles have the same git blob SHA as upstream, so nothing has to be uploaded from the laptop. Every file is re-hashed when the dataset is built or restored.
- **Reported model = `last.pt` after a fixed 20 epochs.** No epoch selection on any split; `best.pt` is ignored. Val is used for monitoring and for pre-declared decisions only (e.g. image size). Test is read only for reporting.
- **Image size (Q2):** the Phase 2 baseline runs at both 640 and 256 px, 3 seeds each. The size used for Phase 3–4 is chosen on **val** Healthy AP@50 (mean over seeds) and logged here before any test number is looked at. Reason: 640 upsamples 256 px tiles, and the old 640 > 256 gap is one seed on a leaky split.
- **Framing (Q4):** the YOLOv8n detector stays the main model (continuity with the project). Every run also reports a tile-level confusion matrix (class of the top box per tile, threshold-free), because every box is the whole tile. A YOLOv8-cls readout is optional, final comparison only.
- **Library pins** (`requirements-colab.txt`): ultralytics 8.4.183, diffusers 0.41.0, transformers 5.20.2, accelerate 1.15.0, peft 0.21.2, all ≥ 2 weeks old on 2026-11-07. torch/torchvision are not pinned (Colab's CUDA build is kept) and are recorded in every run JSON.
- **Original recipe made explicit** (`src/ppsd/configs.py`). Checked against the Ultralytics 8.3.29 source: `optimizer=auto` → AdamW, lr0 0.001667, momentum 0.9, warmup_bias_lr 0. Every key exists in the 8.4.183 defaults; the only differences from 8.4 defaults are the intended ones.
- **Deliberate deviations from the original run, none of which change training:** `augment=False` (the original `augment=True` only turned on TTA in the final validation; **confirmed in source**, `validator.py` line 110); `save_period=-1` (`last.pt` is still saved every epoch); `workers=2`.
- **Resume:** the runner passes `resume=<path to last.pt>`. In 8.4, a bare `resume=True` on a non-resumable checkpoint silently starts a new training (`engine/model.py` lines 877–886). A "nothing to resume" assertion (session died after the final epoch) is caught and treated as finished.
- **Smoke runs evaluate on val only**, so the frozen test set is not touched by pipeline tests.
- **Laptop tests:** `tests/test_ppsd.py`, 8 CPU-only tests (split integrity, build/verify/zip round trip on real files, tamper detection, metrics, epoch counting, notebook generation). All pass.
- **Not yet done:** nothing has run on Colab. The code reaches Colab via a `synthetic-study` branch (needs Amine's OK to push) or by uploading the folder to Drive.
