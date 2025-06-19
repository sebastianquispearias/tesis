from pathlib import Path
import cv2
import json
import numpy as np
import random
import csv

# Base directory
base_dir = Path(r"C:\Users\User\Desktop\tesis\data")

# VT dataset paths
vt_base = base_dir / "Corrosion Condition State Classification" / "512x512"
vt_train_imgs = vt_base / "Train" / "images_512"
vt_train_masks = vt_base / "Train" / "mask_512"
vt_test_imgs  = vt_base / "Test"  / "images_512"
vt_test_masks = vt_base / "Test"  / "mask_512"

# Ameli dataset base
ameli_base    = base_dir / "corrosion images" / "corrosion images"

# Output processed directories
proc_images = base_dir / "processed_binary_384" / "images"
proc_masks  = base_dir / "processed_binary_384" / "masks"
proc_images.mkdir(parents=True, exist_ok=True)
proc_masks.mkdir(parents=True, exist_ok=True)

# Convert JSON/TXT annotation to binary mask
def json_to_mask(json_file, size=(512,512)):
    try:
        data = json.load(open(json_file, 'r'))
    except Exception:
        return None
    mask = np.zeros(size, dtype=np.uint8)
    for shape in data.get("shapes", []):
        pts = np.array(shape.get("points", []), dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask

# Process VT train/test
def process_vt(img_dir, mask_dir, suffix):
    total = 0
    for img_file in sorted(img_dir.glob("*")):
        ext = img_file.suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png"}:
            continue
        mask_file = mask_dir / f"{img_file.stem}.png"
        if not mask_file.exists():
            continue
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        img_res  = cv2.resize(img,       (384,384), interpolation=cv2.INTER_LINEAR)
        mask_res = cv2.resize(mask_bin,  (384,384), interpolation=cv2.INTER_NEAREST)
        out_name = f"{img_file.stem}_{suffix}.png"
        cv2.imwrite(str(proc_images / out_name), img_res)
        cv2.imwrite(str(proc_masks  / out_name), mask_res)
        total += 1
    print(f"VT {suffix}: {total} images processed")

# Process VT
process_vt(vt_train_imgs, vt_train_masks, "vt_train")
process_vt(vt_test_imgs,  vt_test_masks,  "vt_test")

# Process Ameli
def process_ameli(ameli_base_path):
    img_root = ameli_base_path / "images"
    json_root = ameli_base_path / "Annotation json format"
    txt_root  = ameli_base_path / "Annotation txt format"

    # Detectar fases (train, test, Validation)
    phases = [d.name for d in img_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    print(f"[Ameli] Phases: {phases}")

    total = 0
    for phase in phases:
        # Carpeta de la fase (por ejemplo ".../images/train")
        phase_folder = img_root / phase

        # **Nuevo**: dentro de phase_folder hay otra carpeta "images"
        subdirs = [d for d in phase_folder.iterdir() if d.is_dir() and d.name.lower() == "images"]
        if not subdirs:
            print(f"[Ameli][{phase}] No inner 'images' folder found.")
            continue
        img_dir = subdirs[0]

        # Máscaras JSON bajo: .../Annotation json format/phase/label
        json_dir = json_root / phase / "label"
        # Máscaras TXT bajo:  .../Annotation txt format/phase/labels
        txt_dir  = txt_root  / phase / "labels"

        print(f"[Ameli][{phase}] img_dir={img_dir.exists()}, json_dir={json_dir.exists()}, txt_dir={txt_dir.exists()}")

        # Recorremos **directamente** los archivos de imagen
        for img_file in sorted(img_dir.glob("*.jpeg")) + sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
            stem = img_file.stem
            print(f"[Ameli][{phase}] Processing: {img_file.name}")

            # Primero intentamos JSON
            mask0 = None
            jp = json_dir / f"{stem}.json"
            if jp.exists():
                mask0 = json_to_mask(jp)
            else:
                # Si no hay JSON, intentamos TXT
                tp = txt_dir / f"{stem}.txt"
                if tp.exists():
                    mask0 = json_to_mask(tp)

            if mask0 is None:
                print(f"[Ameli][WARNING] No mask found for {stem} in {phase}")
                continue

            # Contar píxeles blancos para verificar la máscara
            whites = int((mask0 > 0).sum())
            print(f"[Ameli][{phase}] Mask whites: {whites}")

            # Leer y redimensionar imagen
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"[Ameli][ERROR] Cannot read image {img_file.name}")
                continue
            img_res = cv2.resize(img, (384,384), interpolation=cv2.INTER_LINEAR)

            # Umbralizar y redimensionar máscara
            _, mask_bin = cv2.threshold(mask0, 127, 255, cv2.THRESH_BINARY)
            mask_res = cv2.resize(mask_bin, (384,384), interpolation=cv2.INTER_NEAREST)

            # Guardar con sufijo indicando fase
            out_name = f"{stem}_ameli_{phase.lower()}.png"
            cv2.imwrite(str(proc_images / out_name), img_res)
            cv2.imwrite(str(proc_masks  / out_name), mask_res)
            total += 1

    print(f"[Ameli] Total processed: {total}")


# Execute Ameli
process_ameli(ameli_base)

# Create splits
def create_splits(image_dir, mask_dir):
    imgs = sorted(image_dir.glob("*.png"))
    random.seed(42)
    random.shuffle(imgs)
    N = len(imgs)
    n_tr = int(0.7 * N)
    n_va = int(0.15 * N)
    splits = {
        "train": imgs[:n_tr],
        "val":   imgs[n_tr:n_tr+n_va],
        "test":  imgs[n_tr+n_va:]
    }
    for split, lst in splits.items():
        with open(base_dir / f"{split}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for img in lst:
                m = mask_dir / img.name
                if not m.exists():
                    continue
                writer.writerow([str(img), str(m)])
    print("Splits generated")

create_splits(proc_images, proc_masks)
print("Dataset ready in processed_binary_384")

