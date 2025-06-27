from pathlib import Path
import cv2
import json
import numpy as np
import random
import csv

# ——— CONFIG ———
base_dir    = Path(r"C:\Users\User\Desktop\tesis\data")
vt_base     = base_dir / "Corrosion Condition State Classification" / "512x512"
ameli_base  = base_dir / "corrosion images" / "corrosion images"

proc_images = base_dir / "processed_binary_384" / "images"
proc_masks  = base_dir / "processed_binary_384" / "masks"
proc_images.mkdir(parents=True, exist_ok=True)
proc_masks.mkdir(parents=True, exist_ok=True)

# ——— Helpers ———
def json_to_mask(json_file, size):
    """Dibuja los polígonos del JSON en una máscara de 'size'."""
    data = json.load(open(json_file, 'r'))
    mask = np.zeros(size, dtype=np.uint8)
    for shape in data.get("shapes", []):
        pts = np.array(shape.get("points", []), dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask

# ——— Procesar VT ———
def process_vt(img_dir, mask_dir, suffix):
    imgs = sorted(img_dir.glob("*"))
    count = 0

    for img_path in imgs:
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        mask_path = mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            continue

        # Leer imagen y máscara original
        img  = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        # 1) Redimensionar imagen
        img_res = cv2.resize(img, (384,384), interpolation=cv2.INTER_LINEAR)

        # 2) Binarizar directamente en la escala original, luego redimensionar
        _, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        mask_res   = cv2.resize(mask_bin, (384,384), interpolation=cv2.INTER_NEAREST)

        # 3) Guardar
        out_name = f"{img_path.stem}_{suffix}.png"
        cv2.imwrite(str(proc_images / out_name), img_res)
        cv2.imwrite(str(proc_masks  / out_name), mask_res)

        count += 1

    print(f"VT {suffix}: {count} imágenes procesadas")



# Ejecutar VT
process_vt(vt_base / "Train" / "images_512", vt_base / "Train" / "mask_512", "vt_train")
process_vt(vt_base / "Test"  / "images_512", vt_base / "Test"  / "mask_512", "vt_test")

# ——— Procesar Ameli ———
def process_ameli(ameli_base):
    img_root  = ameli_base / "images"
    json_root = ameli_base / "Annotation json format"
    txt_root  = ameli_base / "Annotation txt format"

    phases = [d.name for d in img_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    total = 0

    for phase in phases:
        inner = img_root / phase / "images"  # carpeta interna
        jdir  = json_root / phase / "label"
        tdir  = txt_root  / phase / "labels"
        for img_path in sorted(inner.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # JSON o TXT
            mask0 = None
            jfile = jdir / f"{img_path.stem}.json"
            if jfile.exists():
                mask0 = json_to_mask(jfile, size=(h, w))
            else:
                tfile = tdir / f"{img_path.stem}.txt"
                if tfile.exists():
                    mask0 = json_to_mask(tfile, size=(h, w))

            if mask0 is None:
                continue

            # redimensionar
            img_res  = cv2.resize(img,       (384,384), interpolation=cv2.INTER_LINEAR)
            mask_bin = cv2.resize(mask0,     (384,384), interpolation=cv2.INTER_NEAREST)
            _, mask_res = cv2.threshold(mask_bin, 127, 255, cv2.THRESH_BINARY)
    
            out_name = f"{img_path.stem}_ameli_{phase.lower()}.png"
            cv2.imwrite(str(proc_images / out_name), img_res)
            cv2.imwrite(str(proc_masks  / out_name), mask_res)
            total += 1

    print(f"Ameli total: {total} imágenes procesadas")

process_ameli(ameli_base)

# ——— Crear splits (70/15/15) ———
def create_splits(img_dir, mask_dir):
    all_imgs = sorted(img_dir.glob("*.png"))
    random.seed(42)
    random.shuffle(all_imgs)
    N = len(all_imgs)
    n1 = int(0.7 * N)
    n2 = int(0.15 * N)
    splits = {
        "train": all_imgs[:n1],
        "val":   all_imgs[n1:n1+n2],
        "test":  all_imgs[n1+n2:]
    }
    for name, lst in splits.items():
        with open(base_dir / f"{name}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            for ip in lst:
                mp = mask_dir / ip.name
                if mp.exists():
                    wr.writerow([str(ip), str(mp)])
    print("CSV splits generados")

create_splits(proc_images, proc_masks)

print("¡Listo! Dataset unificado en 'processed_binary_384' y CSVs en la carpeta base.")
