# Phase 0: Understanding the existing project

Status: read-only audit, no project code changed. Written 2026-11-07.
Evidence: every count below comes from `phase0/audit_dataset.py` (output in `phase0/audit_output.json`), `phase0/capture_groups.py`, the saved run folders, or the notebooks' saved outputs. Where something is inferred rather than read, it says so.

---

## 0. Summary

1. **This is tile classification dressed as detection.** Every label file contains one box, `class 0.5 0.5 1.0 1.0`, which covers the whole image. That is why mAP@50-95 ≈ mAP@50 (0.630 vs 0.633). The class decides the score; the box geometry does not.
2. **The validation and test sets leak heavily from train.** Tiles were split at random, one tile at a time, so tiles from the *same photo* land in different splits. 97% of val tiles and 97% of test tiles come from a photo that also has tiles in train, and most of them sit next to a train tile. There are also 22 groups of byte-identical images across splits (the same photo stored under two names).
3. **The weak class is Healthy, and it is very small:** 86 train / 18 val / 19 test tiles, from 65 distinct photos in total. One healthy val tile = 5.6 recall points.
4. **The reported metrics mostly reflect Healthy failing.** In the 640 px run, the saved confusion matrix shows **1 of 18** healthy val tiles predicted as Healthy. On the test split, the 256 px model and the MobileNetV2 classifier both get **0 of 19** healthy tiles right. Both predict "Diseased" for everything; MobileNet's 96.13% test accuracy is exactly the share of Diseased tiles (472/491).
5. **The "four runs" are two experiments with one seed each.** `train`, `train2` and `train3` have identical metrics in every epoch (same seed, deterministic); `train2` stopped after 3 epochs. Only `train3` (640 px) and `train4` (256 px) differ.
6. **The dataset is public and MIT-licensed:** the CactiViT dataset (Berka et al., 2023). The README does not credit it yet.

These points change Phase 2–4, so I need answers to the questions in section 9 before starting Phase 1.

---

## 1. Folder layout

| Path | Tracked in git? | What it is |
|---|---|---|
| `README.md` | yes | Project description and results table |
| `YoLo.ipynb` | yes | Split helper (not run here, call commented out), whole-image label writer, `data.yaml` writer, YOLOv8n training (saved output is the `train4` / 256 px run), test-set classification report |
| `Mobilenet.ipynb` | yes | Runs the split into `DatasetPreped/`, trains a frozen MobileNetV2 + small head twice (without/with Keras augmentation), evaluates on the test split |
| `realtime.py` | yes | Webcam loop; loads `datasets/runs/detect/train4/weights/last.pt`, camera index 1 |
| `requirements.txt` | yes | A `pip freeze` of the Windows venv (ultralytics 8.3.29, TF 2.18, many unrelated packages). torch is **not** in it; the notebook reports torch 2.5.1+cu118 |
| `yolov8n.pt` | yes | COCO-pretrained YOLOv8n base weights |
| `datasets/runs/detect/train` | yes | 640 px, 20 epochs, seed 0 |
| `datasets/runs/detect/train2` | yes | Same config, stopped after 3 epochs |
| `datasets/runs/detect/train3` | yes | Same config as `train`; identical metrics (the README's 640 px result) |
| `datasets/runs/detect/train4` | yes | 256 px, 20 epochs, seed 0 (the README's 256 px result; used by `realtime.py`) |
| `datasets/runs/detect/val` | yes | A validation of the **untrained COCO** `yolov8n.pt`: its PR curve is labelled "person 0.001 / bicycle 0.074, mAP 0.038". Not a trained model; ignore it |
| `DatasetPreped/{train,val,test}/{Healthy,Diseased}` | **no** (untracked, 430 MB) | Output of the split, used by MobileNet |
| `DatasetPreped2/images/...`, `labels/...`, `data.yaml` | **no** (untracked, 436 MB) | YOLO layout. The images are byte-identical to `DatasetPreped` (3252/3252 md5 match) |

Run folders keep a checkpoint for every epoch (`save_period=1`). The git pack is 769 MB, mostly those `.pt` files.

The original working directory was `C:\Users\khali\OneDrive\Bureau\amine\CactiViT-materials-master\` on an RTX 4070 Laptop GPU (8 GB), Windows, Python 3.10.11. So training happened on a team member's laptop, not on Colab.

---

## 2. Dataset

### 2.1 Origin
- `YoLo.ipynb` and `Mobilenet.ipynb` read from `Datasets/DBX/Cactus/{Healthy, Damaged, EarlyStage, LateStage, Old_Dead}`.
- This matches the public repository **AnasBerka/CactiViT-materials** (`Datasets/DBX/...`), the dataset of *Berka et al., "CactiViT: Image-based smartphone application and transformer network for diagnosis of cactus cochineal", Artificial Intelligence in Agriculture 9 (2023) 12–21, doi:10.1016/j.aiia.2023.07.002*. I checked this through the GitHub API on 2026-11-07:
  - upstream folder counts: Healthy 123, Damaged 100, EarlyStage 301, LateStage 2625, Old_Dead 103, Confused 2038, NoCactus 668;
  - all 123 local Healthy file names are in upstream `Healthy`, and all 3129 local Diseased file names are in the four upstream disease folders. So the local data is exactly 5 of the 7 upstream folders, with nothing added. `Confused` and `NoCactus` are unused.
- The "disease" is **cochineal infestation** (*Dactylopius opuntiae*, visible as white waxy colonies), not a pathogen. "Diseased" merges four stages: Damaged, EarlyStage, LateStage, Old_Dead.

### 2.2 What an image is
Each file is a **tile cut from a larger smartphone photo**. The file name encodes the source and the crop: `<source>_<y0>_<y1>_<x0>_<x1>.png`.
- 1024 px crops (from 4000×5000-ish photos named `img_YYYYMMDD_HHMMSS`) were downscaled to 256 px.
- 256 px crops (from smaller photos named by UUID or `img-YYYYMMDD-waNNNN`, i.e. WhatsApp copies) are stored at native size.

All images are RGB PNG.

### 2.3 Counts per split and class (DatasetPreped2)

| Split | Healthy tiles | Diseased tiles | Total | Boxes | Healthy source photos |
|---|---:|---:|---:|---:|---:|
| train | 86 | 2189 | 2275 | 2275 | 54 |
| val | 18 | 468 | 486 | 486 | 16 |
| test | 19 | 472 | 491 | 491 | 18 |
| **all** | **123** | **3129** | **3252** | **3252** | **65** distinct |

- Healthy is **3.8%** of tiles; the imbalance is about 1 : 25.
- Tile sizes: 256×256 for most; 256×208 for 120 tiles and 132×256 for 30 tiles (image-edge tiles, all Diseased).
- No missing, empty or orphan label files. Every class id matches its folder.
- Only tiles exist, both here and upstream (the upstream file names are the same tile names); the full-resolution source photos are not available.

### 2.4 Labels and box statistics
- Format: YOLO txt, `class x_center y_center width height`, normalised. Class 0 = Healthy, 1 = Diseased (`data.yaml`, `nc: 2`).
- **Every one of the 3252 boxes is `0.5 0.5 1.0 1.0`**: one box per image, covering 100% of it, for both classes. There are no other box sizes, so per-class box statistics are trivial (width = height = 1.0).
- The boxes were not annotated; `YoLo.ipynb` cell 6 writes them automatically from the folder name ("Write bounding box data for the whole image").
- Consequence: the model learns "which class is this whole tile", and IoU with the label is almost always high. mAP@50-95 ≈ mAP@50 in every epoch confirms this.

### 2.5 Content (visual check of 16 random train tiles)
Healthy tiles show green pads, but some show mostly sky, scrub or rocks with a small piece of cactus. Diseased tiles show white cochineal wax, dried or collapsed pads, and some show mostly background as well. The tile labels come from the upstream labelling and are not checked per tile here.

---

## 3. Exact training setup (from `args.yaml` and the saved notebook output)

| Item | Value |
|---|---|
| Model | `yolov8n.pt` (COCO-pretrained), 3.01 M params, `nc` overridden to 2 |
| Framework | Ultralytics 8.3.29, torch 2.5.1+cu118, Python 3.10.11, RTX 4070 Laptop 8 GB |
| Image size | 640 (`train`, `train2`, `train3`), 256 (`train4`). **The tiles are ≤256 px, so 640 means upsampling 2.5×** |
| Epochs / batch | 20 / 16, `patience=100`, so no early stop |
| Optimizer | `auto`, which resolved to AdamW lr=0.001667, momentum 0.9 (logged for `train4`; same rule applies to `train3`). `lr0`/`momentum` in args are ignored |
| Schedule | linear LR decay, `lrf=0.01`, 3 warm-up epochs, mosaic closed for the last 10 epochs |
| Augmentation | hsv_h 0.015, hsv_s 0.7, hsv_v 0.4, translate 0.1, scale 0.5, fliplr 0.5, mosaic 1.0, mixup 0, copy_paste 0, no rotation/shear/perspective (`auto_augment`/`erasing` are classification-only settings) |
| Seed | `seed=0`, `deterministic=True`, AMP on |
| Class balancing | none |
| "Best epoch" | Ultralytics `best.pt` = epoch with highest fitness = 0.1·mAP@50 + 0.9·mAP@50-95 **on the val split**. The README's "best epoch" rows (19 and 17) are those epochs |
| What val was used for | per-epoch monitoring, choosing `best.pt`, the README's chosen numbers, and the 640-vs-256 decision. **All on the same 486 val tiles** |
| What test was used for | only a `classification_report` cell at the end of `YoLo.ipynb` and in `Mobilenet.ipynb`; never in the README |
| `augment=True` | passed to `train()`. In Ultralytics this flag means test-time augmentation for validation/prediction. **Confirmed in the 8.3.29 source** (`engine/validator.py` line 110: `augment = self.args.augment and (not self.training)`): it is off during per-epoch validation and **on during the final validation of `best.pt`**. That is why the final numbers differ from the per-epoch row (see section 6). |

The cell that trained `train3` is not in the saved notebook (it was overwritten by the `train4` run). The 640 px setup is known only from `train3/args.yaml`.

### Bugs and oddities found while reading
- `YoLo.ipynb` cell 11 loads `model_test` from `train4/weights/last.pt`, but cell 13 predicts with `model` (the object left in memory by training), so `model_test` is never used. Result on test: Healthy 0/19, Diseased 472/472.
- `Mobilenet.ipynb`: MobileNetV2 expects inputs scaled to [-1, 1], but the dataset feeds raw 0–255 pixels; no `Rescaling`/`preprocess_input` layer is in the model. There are no class weights. Both trained heads predict Diseased for every image: val accuracy 0.9630 = 468/486, test 0.9613 = 472/491, confusion matrix `[[472, 0], [19, 0]]`. TensorFlow did not see the GPU (CPU training).
- `realtime.py` uses `train4/last.pt` (epoch 20). The final validation reports Healthy AP@50 = 0.094 for `train4`'s `best.pt`.

---

## 4. Near-duplicates and leakage between splits

Method: parse the source-photo id from each file name; md5 hash for exact duplicates; 8×8 difference-hash and 16×16 thumbnail cosine similarity for near-duplicates; tile coordinates for spatial adjacency; capture timestamps for bursts. Everything is in `phase0/`.

| Check | val (486 tiles) | test (491 tiles) |
|---|---:|---:|
| Tiles whose **source photo also has tiles in train** | 470 (96.7%) | 477 (97.1%) |
| …Healthy only | 18 / 18 (100%) | 17 / 19 (89.5%) |
| Tiles **touching a train tile** of the same photo | 452 | 457 |
| Tiles with a pixel-near-duplicate in train (dHash ≤ 4, cosine ≥ 0.95) | 16 | 18 |

- **Exact duplicates:** 61 groups of byte-identical images, 22 of them across splits, none across classes. They come from 8 WhatsApp copies (`img-20210327-waNNNN`) that duplicate UUID-named photos. For example, `val/.../46cc5048-…_0_256_1024_1280.png` is identical to `train/.../img-20210327-wa0070_0_256_1024_1280.png`. All 34 pixel-near-duplicates above are these copies.
- **Photo-level overlap:** 441 source photos in total; 250 appear in both train and val, 261 in both train and test, 176 in all three splits. 56 photos have tiles in both classes (part of the plant infested, part not).
- **Capture bursts:** 275 photos have timestamps, all from two days (2021-03-27 and 2021-05-23). Grouping photos taken ≤10 s apart gives 54 bursts; ≤30 s gives 20. Every timestamped val/test tile shares a burst with a train tile. The 37 timestamped Healthy photos fall into 20 bursts at ≤10 s.
- **Effective healthy sample size:** 123 healthy tiles → 65 photos → roughly 20 bursts + 28 untimed photos. The number of truly independent healthy examples is **at most about 50**.

**Conclusion:** the current val/test numbers measure recognition of tiles from photos the model has already seen, not generalisation to new plants or photos. Even so, Healthy is poorly recognised, which says the problem is genuinely hard or the class is too small.

---

## 5. Licence and terms

- **Dataset:** the CactiViT-materials repository carries an **MIT License** ("Copyright (c) 2022 Anas BERKA"; file `licence`, GitHub API reports `mit`). MIT allows use, modification, publication and distribution, provided the copyright and permission notice are kept. As I read it, this covers training, generating derived/synthetic images, and publishing results. Caveats:
  - MIT is written for "software and documentation", and applying it to a photo dataset is a little unusual. I am not a lawyer; if in doubt, the authors can be asked by email.
  - The paper should be cited and the notice kept with any shared derived images.
  - The project README currently does **not** credit the dataset source. That should be fixed (a later, separate change; I won't touch the README without your approval).
- **IEEE TUNED 7.0 competition terms:** nothing in the repository describes them. **I cannot tell whether the rules restrict reusing or publishing the competition work. Please check.**
- **Team project:** the runs were made on a team member's machine. For a resume, the original detector should be described as a team result, and the new study as your own extension. **Please confirm the team is fine with you extending and publishing it.**
- **Stable Diffusion 1.5 / inpainting checkpoint licences:** to be checked in Phase 3, as planned.

---

## 6. README numbers vs what the saved runs show

| README claim | Saved evidence | Verdict |
|---|---|---|
| 640 px, best epoch 19: P 0.6859, R 0.6389, mAP@50 0.6328, mAP@50-95 0.6303 | `train3/results.csv` row 19: identical | ✔ matches; chosen on val, reported on val |
| 256 px, best epoch 17: P 0.9814, R 0.5000, mAP@50 0.5674, mAP@50-95 0.5659 | `train4/results.csv` row 17: 0.98139 / 0.5 / 0.56735 / 0.56591 | ✔ matches |
| Class AP@50: Healthy 0.356, Diseased 0.988, all 0.672 | `train3/PR_curve.png` | ✔ matches. This is the final validation of `best.pt` **with test-time augmentation** (`augment=True`, confirmed in source), which is why it differs from row 19 (0.633) |
| 256 px final validation (not in README) | notebook output: all 0.537, **Healthy P 0.037 / AP 0.094**, Diseased AP 0.98 | — |
| "Four tracked runs" (your summary) | `train` = `train3` metric-for-metric; `train2` = their first 3 epochs | ✘ really 2 configurations × 1 seed |
| "≈0.69 precision, 0.64 recall" at 640 | overall P/R are **means over the two classes**, at one confidence threshold | ✔ numerically, but misleading. With Diseased recall near 1, mean recall 0.639 implies Healthy recall ≈ 0.28 at that threshold (inferred). The confusion matrix at Ultralytics' default threshold shows 1/18 |
| "At 256 px recall dropped to about 0.50" | R = 0.500 exactly from epoch 11 to 19 | ✔. 0.5 is what you get when Diseased recall = 1 and Healthy recall = 0. The 0.98 "precision" next to it does **not** mean the model is precise; it comes from how Ultralytics averages precision when a class has (almost) no detections |
| "Diseased detected much better than Healthy" | AP 0.988 vs 0.356 | ✔. Diseased AP is near-trivial here: 96% of tiles are Diseased, every box is the full tile, and tiles leak from train |
| MobileNetV2 classifier | 96.13% test accuracy = always predicting Diseased | ✘ not a working classifier |

**Per-epoch instability:** Healthy contributes in steps of 1/18. Mean recall jumps between 0.50 and 0.97 from one epoch to the next (`train3` epochs 3, 7, 11), so a "best epoch" chosen on val partly picks noise.

---

## 7. Limits of the original evaluation, and a proposed fix

The limits, in order of size:
1. **Leakage** (section 4): val/test share photos, bursts and even identical files with train.
2. **Selection on the reporting set:** `best.pt`, the "best epoch" and the 640-vs-256 choice were all picked on the same val set that is reported. The numbers are therefore optimistic.
3. **Tiny Healthy evaluation set:** 18 tiles from 16 photos; one tile = 5.6 recall points.
4. **One seed per configuration:** no spread can be estimated.

The fix in the brief (split the current val into a val part for epoch choice and a held-out test part) addresses limit 2 only. It also halves the Healthy evaluation set to about 9 tiles (≈11 points per tile) and keeps the leakage. There is already an unused test split, but it leaks too. So I propose three options and recommend **B**:

| Option | What | Fixes | Cost / trade-off |
|---|---|---|---|
| **A. Minimal** | Keep the original split. Choose the epoch on val, report on the existing test split (19 healthy) | selection bias | Still leaked. Numbers stay comparable to the old ones but remain optimistic |
| **B. Grouped re-split (recommended)** | New train / val / test split **by group**, where a group = source photo + its exact-duplicate twin + photos ≤10 s apart. Stratified so Healthy groups are spread (~60/20/20 by groups). **Train a fixed 20 epochs and report `last.pt`**, so no epoch is chosen at all. Val is used only for sanity checks. Test is frozen and real-only | leakage + selection bias | Not comparable to README numbers (old numbers kept, labelled "leaked split"). Healthy test ≈ 20–30 tiles. One LoRA and one synthetic pool |
| **C. Grouped 5-fold CV** | Same groups, 5 folds; each fold is the test set once | leakage + selection + uses all 123 healthy tiles for evaluation | ~5× the training cost, and the LoRA and synthetic images must be rebuilt per fold (training split only). Heavy on a free T4; better as a follow-up if B is borderline |

With B, Phase 2's baseline X = the original recipe retrained on the new split, with 3 seeds (I suggest 5 if 256 px is cheap enough). The old leaked numbers stay in the README for history.

---

## 8. What this means for the Stable Diffusion plan (to discuss, not decided)

- **Box validity is trivial; class validity is the real risk.** Since the box is the whole tile, any generated tile keeps a valid box. The thing to verify is whether a "Healthy" synthetic tile really contains no cochineal (and is not just sky). Method A (inpaint inside the box) becomes "regenerate most of the tile", and method B (img2img at low strength) becomes "perturb a real healthy tile". The 30-sample check with boxes drawn should be replaced, or extended, by a check of **class correctness**.
- **Target class:** Healthy (86 train tiles from 54 photos). This is also the LoRA training set, which is very small.
- **Statistical power:** with about 20–30 healthy test tiles, an improvement must be large (several tiles) to beat seed spread. A clean "not distinguishable" result is likely and acceptable.
- **Native resolution is 256 px** and SD 1.5 works at 512 px. Generated tiles will be downscaled, which is fine, but I'd record it.
- **Honest wording:** the resume line should not say "object detector" without qualification, since every box is a whole tile. Possible wording: "YOLOv8 tile-level detector (whole-tile boxes)". Alternatively, add a YOLOv8-cls readout next to the detector.
- **Unused upstream data:** `NoCactus` (668 tiles) could serve as background/negative tiles, and the four diseased sub-stages allow per-stage error analysis (e.g. is Healthy confused mostly with EarlyStage?). I suggest only the per-stage analysis (free) and leaving `NoCactus` out to keep the scope.
- **Compute:** on the 4070 laptop, 20 epochs took about 6 min at 640 px and 3.5 min at 256 px. A T4 will be slower; I will measure it in Phase 1 rather than guess.

---

## 9. Questions I could not answer from the repository

1. **Evaluation protocol:** adopt **B** (grouped re-split, fixed epochs, frozen real test)? Or A or C? Is ≤10 s the right burst threshold, or do you know which photos show the same plant?
2. **Baseline image size:** the README's headline uses 640 px (upsampled from 256). Should X be 640, 256, or both?
3. **Competition rules:** do the IEEE TUNED 7.0 terms say anything about reusing or publishing the work or the data?
4. **Team:** is the team OK with you extending and publishing this? Who ran `train`–`train4` (machine user `khali`)? How should the original work be credited in the report and on your resume?
5. **Dataset licence:** are you comfortable relying on the MIT licence of CactiViT-materials, with a citation? Do you want me to add the dataset attribution to the README (a separate small change)?
6. **Framing:** keep the YOLO detector for continuity (and say "whole-tile boxes" honestly), or also add a YOLOv8-cls classifier readout?
7. **Labels:** do you know how the upstream authors labelled tiles (per tile by an expert, or per photo)? 56 photos have tiles in both classes, so it looks per-tile.
8. **Original photos:** do you have the full-resolution source photos? They are not in the repo; tiles would be enough, but photos would allow other crops.
9. **Google Drive:** how much free space? The YOLO dataset is about 0.44 GB, SD 1.5 weights are about 2–4 GB, and synthetic images plus checkpoints come on top.
10. **Original code:** confirm I should leave `YoLo.ipynb`, `Mobilenet.ipynb`, `realtime.py` and the README untouched (the bugs above are only documented, not fixed).

---

Sources checked online (2026-11-07): https://github.com/AnasBerka/CactiViT-materials (README, `licence`, folder tree via the GitHub API); paper DOI 10.1016/j.aiia.2023.07.002 (as cited in that README; full text not read).
