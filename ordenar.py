import csv
import shutil
from pathlib import Path

# ——— RUTAS ———
base_dir       = Path(r"C:\Users\User\Desktop\tesis\data")
processed_base = base_dir / "processed_binary_384"
csv_files = {
    "Train": base_dir / "train.csv",
    "Val":   base_dir / "val.csv",
    "Test":  base_dir / "test.csv",
}

# ——— CREAR CARPETAS ———
for split in csv_files:
    (processed_base / split / "images").mkdir(parents=True, exist_ok=True)
    (processed_base / split / "masks").mkdir(parents=True, exist_ok=True)

# ——— COPIAR ARCHIVOS SEGÚN CSV ———
for split, csv_path in csv_files.items():
    img_out = processed_base / split / "images"
    msk_out = processed_base / split / "masks"
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for img_fp, msk_fp in reader:
            src_img = Path(img_fp)
            src_msk = Path(msk_fp)
            # Copia imagen
            shutil.copy(src_img, img_out / src_img.name)
            # Copia máscara
            shutil.copy(src_msk, msk_out / src_msk.name)
    print(f"{split}: copiados {len(list((img_out).iterdir()))} imágenes y {len(list((msk_out).iterdir()))} máscaras")
