#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_dataset_unified.py – Final preprocessing script (VT‑Bridge + Ameli)
==========================================================================
This script merges the logic of your original **prepare_dataset_final.py** and
our v2 prototype, adding robustness against nested folders (e.g. Train / Test)
and ensuring that **validation and test contain only VT images**.  Ameli images
– labelled or not – are copied but used exclusively in the *train* CSV (for
semi‑supervised experiments).

Workflow
--------
1. Locate every *images_512* directory under ``--src_vt``; expect a sibling
   *mask_512* folder.
2. Read VT images (.png / .jpg / .jpeg), binarise masks (>0 → 255), resize to
   384², save as ``vt_<orig>.png``.
3. Locate Ameli images under
   ``images/PHASE/images/*.png`` (same as your original script). If a JSON/TXT
   annotation exists, convert to mask; otherwise create an empty mask (unlabeled).
4. Write outputs to ``<out>/images`` and ``<out>/masks``.
5. Create CSV splits with seed 42:
     * train.csv = VT‑train + all Ameli images
     * val.csv / test.csv = VT‑val / VT‑test only
   plus sub‑splits ``vt_train_10/25/50.csv`` on the VT‑train subset.

Example
-------
```
python prepare_dataset_unified.py \
  --src_vt    C:/Users/User/Desktop/tesis/data/Corrosion Condition State Classification/512x512 \
  --src_ameli C:/Users/User/Desktop/tesis/data/corrosion images/corrosion images \
  --out       C:/Users/User/Desktop/tesis/data/processed_binary_384
```
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import cv2, numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

IMG_EXTS = {'.png', '.jpg', '.jpeg'}
SIZE = (384, 384)

def ensure(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def resize(img: np.ndarray, size=SIZE, interp=cv2.INTER_LINEAR):
    return cv2.resize(img, size, interpolation=interp)

def json_to_mask(p: Path, shape):
    data = json.load(p.open('r', encoding='utf-8'))
    mask = np.zeros(shape, np.uint8)
    for shp in data.get('shapes', []):
        pts = np.array(shp['points'], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask

def txt_to_mask(p: Path, shape):
    pts = []
    for line in p.read_text().splitlines():
        if ',' in line:
            x, y = map(float, line.split(','))
            pts.append([int(x), int(y)])
    mask = np.zeros(shape, np.uint8)
    if pts:
        cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
    return mask


def save_pair(img: np.ndarray, mask: np.ndarray, stem: str, prefix: str, out_imgs: Path, out_masks: Path):
    fname = f"{prefix}{stem}.png"
    cv2.imwrite(str(out_imgs  / fname), resize(img))
    cv2.imwrite(str(out_masks / fname), resize(mask, interp=cv2.INTER_NEAREST))
    return fname

# -----------------------------------------------------------------------------
# VT processing (recursive)
# -----------------------------------------------------------------------------

def collect_vt(src_vt: Path, out_imgs: Path, out_masks: Path):
    vt_files = []
    for img_dir in src_vt.rglob('images_512'):
        mask_dir = img_dir.parent / 'mask_512'
        if not mask_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            mask_path = mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                continue
            img  = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            _, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
            out_name = save_pair(img, mask_bin, img_path.stem, 'vt_', out_imgs, out_masks)
            vt_files.append(out_name)
    print(f'[VT] saved {len(vt_files)} images')
    return vt_files

# -----------------------------------------------------------------------------
# Ameli processing
# -----------------------------------------------------------------------------

def collect_ameli(src_ameli: Path, out_imgs: Path, out_masks: Path):
    img_root  = src_ameli / 'images'
    json_root = src_ameli / 'Annotation json format'
    txt_root  = src_ameli / 'Annotation txt format'
    am_files = []
    for phase_dir in img_root.glob('*'):
        inner = phase_dir / 'images'
        if not inner.exists():
            continue
        phase = phase_dir.name.lower()
        for img_path in inner.iterdir():
            if img_path.suffix.lower() != '.png':
                continue
            img = cv2.imread(str(img_path))
            H, W = img.shape[:2]
            j = json_root / phase_dir.name / 'label' / f"{img_path.stem}.json"
            t = txt_root  / phase_dir.name / 'labels' / f"{img_path.stem}.txt"
            if j.exists():
                mask0 = json_to_mask(j, (H, W))
            elif t.exists():
                mask0 = txt_to_mask(t, (H, W))
            else:
                mask0 = np.zeros((H, W), np.uint8)
            _, mask_bin = cv2.threshold(mask0, 127, 255, cv2.THRESH_BINARY)
            out_name = save_pair(img, mask_bin, f"{img_path.stem}_{phase}", 'ameli_', out_imgs, out_masks)
            am_files.append(out_name)
    print(f'[Ameli] saved {len(am_files)} images')
    return am_files

# -----------------------------------------------------------------------------
# Split helpers
# -----------------------------------------------------------------------------

def write_csv(path: Path, names):
    path.write_text('\n'.join(names), encoding='utf-8')

def create_splits(vt_files, am_files, out_dir: Path):
    random.shuffle(vt_files)
    N = len(vt_files)
    n_train, n_val = int(0.7*N), int(0.15*N)
    vt_train = vt_files[:n_train]
    vt_val   = vt_files[n_train:n_train+n_val]
    vt_test  = vt_files[n_train+n_val:]
    train = vt_train + am_files
    write_csv(out_dir/'train.csv', train)
    write_csv(out_dir/'val.csv', vt_val)
    write_csv(out_dir/'test.csv', vt_test)
    print(f"[SPLIT] Train {len(train)} (VT {len(vt_train)} + Ameli {len(am_files)}) | Val {len(vt_val)} | Test {len(vt_test)}")
    for pct in (0.10, 0.25, 0.50):
        k = int(len(vt_train)*pct)
        write_csv(out_dir/f'vt_train_{int(pct*100)}.csv', vt_train[:k])

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_vt',    type=Path, required=True, help='VT 512x512 root')
    ap.add_argument('--src_ameli', type=Path, required=True, help='Ameli root (corrosion images)')
    ap.add_argument('--out',       type=Path, required=True, help='Output folder')
    args = ap.parse_args()

    img_out = args.out / 'images'
    msk_out = args.out / 'masks'
    ensure(img_out); ensure(msk_out)

    vt_files = collect_vt(args.src_vt, img_out, msk_out)
    am_files = collect_ameli(args.src_ameli, img_out, msk_out)
    create_splits(vt_files, am_files, args.out)
    print('Done ✔')

if __name__ == '__main__':
    main()
