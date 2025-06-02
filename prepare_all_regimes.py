#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_all_regimes.py – Preprocesamiento + creación de carpetas por régimen

Este script combina la lógica de:
 - Colectar y preprocesar VT (resize 384×384, binarización, prefijo vt_)
 - Colectar y preprocesar Ameli (resize 384×384, binarización, prefijo ameli_)
 - Generar CSVs de partición (train.csv, val.csv, test.csv, vt_train_10/25/50)
 - Crear carpetas por régimen A-Full, B10–B75, C10–C75 copiando desde “processed/”

Estructura de salida en OUT_ROOT:
  ├── processed/
  │   ├── images/     ← imágenes (384×384) prefijadas: vt_* y ameli_*
  │   ├── masks/      ← máscaras (384×384) binarizadas: vt_* y ameli_*
  │   ├── train.csv
  │   ├── val.csv
  │   ├── test.csv
  │   ├── vt_train_10.csv
  │   ├── vt_train_25.csv
  │   └── vt_train_50.csv
  └── regimes/
      ├── A-Full/
      │   ├── train/
      │   │    ├── images_labeled/
      │   │    ├── masks_labeled/
      │   │    └── images_unlabeled/  (vacío)
      │   ├── val/
      │   │    ├── images/
      │   │    └── masks/
      │   └── test/
      │        ├── images/
      │        └── masks/
      ├── B10/  (B25, B50, B75 idéntico estructura)
      │   ├── train/
      │   │    ├── images_labeled/
      │   │    ├── masks_labeled/
      │   │    └── images_unlabeled/  (vacío)
      │   ├── val/
      │   │    ├── images/
      │   │    └── masks/
      │   └── test/
      │        ├── images/
      │        └── masks/
      └── C10/  (C25, C50, C75 idéntico)
          ├── train/
          │    ├── images_labeled/
          │    ├── masks_labeled/
          │    └── images_unlabeled/  (VT-restante + Ameli)
          ├── val/
          │    ├── images/
          │    └── masks/
          └── test/
               ├── images/
               └── masks/
"""

import argparse
import random
import json
from pathlib import Path
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

# —────────────────────────────────────────────────────────────────────────────
# CONSTANTES Y PARÁMETROS
# —────────────────────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Extensiones válidas de imagen
IMG_EXTS = {'.png', '.jpg', '.jpeg'}

# Tamaño final para imágenes y máscaras
TARGET_SIZE = (384, 384)

# Porcentajes para regímenes Bxx y Cxx
WEAK_PERCENTS = [10, 25, 50, 75]
SEMI_PERCENTS = [10, 25, 50, 75]

# Proporciones iniciales para dividir VT completo
VT_VAL_TEST_SPLIT = 0.30  # 30% total (→ 15% val + 15% test)
VAL_RATIO = 0.5          # dentro del 30%, 50% a val (15%) y 50% a test (15%)

# RUTAS DE ENTRADA POR DEFECTO (editar según tu máquina)
VT_ROOT_DEFAULT    = Path(r"C:\Users\User\Desktop\tesis\data\Corrosion Condition State Classification\512x512")
AMELI_ROOT_DEFAULT = Path(r"C:\Users\User\Desktop\tesis\data\images")  # carpeta que contemple /train/images, /validation/images, /test/images

# RUTA DE SALIDA GLOBAL POR DEFECTO (crea “processed” y “regimes” aquí)
OUT_ROOT_DEFAULT   = Path(__file__).parent / "data" / "all_results"

# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dir(p: Path):
    """Crear carpeta recursivamente si no existe."""
    p.mkdir(parents=True, exist_ok=True)

def resize_image(img: np.ndarray, size=TARGET_SIZE, interp=cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(img, size, interpolation=interp)

def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Umbral simple: todo píxel > 0 → 255."""
    _, m_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    return m_bin

def save_image_and_mask(
    img: np.ndarray, mask: np.ndarray,
    out_imgs: Path, out_masks: Path,
    prefix: str, stem: str
) -> str:
    """
    - Redimensiona imagen y máscara a TARGET_SIZE.
    - Binariza la máscara (si no lo estaba).
    - Guarda ambos como '{prefix}{stem}.png'.
    - Retorna el nombre de archivo generado.
    """
    fname = f"{prefix}{stem}.png"

    # Redimensionar y guardar imagen
    img_resized = resize_image(img, TARGET_SIZE, interp=cv2.INTER_LINEAR)
    cv2.imwrite(str(out_imgs / fname), img_resized)

    # Redimensionar y guardar máscara
    mask_bin = binarize_mask(mask)
    mask_resized = resize_image(mask_bin, TARGET_SIZE, interp=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_masks / fname), mask_resized)

    return fname

# ──────────────────────────────────────────────────────────────────────────────
# RECOLECCIÓN Y PREPROCESAMIENTO DE VT
# ──────────────────────────────────────────────────────────────────────────────

def collect_vt(src_vt: Path, proc_imgs: Path, proc_masks: Path) -> list[str]:
    """
    Recorre recursivamente src_vt buscando carpetas 'images_512' manteniendo
    la jerarquía de Train/ o Test/ para evitar solapamiento de nombres.
    Para cada imagen encontrada:
      1) Lee imagen color y su máscara gris (en la carpeta paralela 'mask_512').
      2) Redimensiona ambos a 384×384 y binariza la máscara.
      3) Guarda en proc_imgs y proc_masks con prefijo 'vt_<subset>_'.
    Retorna lista de nombres guardados (ej. 'vt_train_0001.png', 'vt_test_0023.png').
    """
    vt_files = []
    for img_dir in src_vt.rglob("images_512"):
        subset = img_dir.parent.name  # normalmente 'Train' o 'Test'
        mask_dir = img_dir.parent / "mask_512"
        if not mask_dir.exists():
            continue

        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            # Máscara correspondiente
            mask_path = mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                continue

            # Leer
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue

            prefix = f"vt_{subset.lower()}_"    # vt_train_ o vt_test_
            stem = img_path.stem                # e.g. '0001'
            out_name = save_image_and_mask(
                img, mask,
                proc_imgs, proc_masks,
                prefix, stem
            )
            vt_files.append(out_name)

    print(f"[VT] procesadas {len(vt_files)} imágenes (384×384, binarizadas).")
    return vt_files

# ──────────────────────────────────────────────────────────────────────────────
# RECOLECCIÓN Y PREPROCESAMIENTO DE AMELI
# ──────────────────────────────────────────────────────────────────────────────

def collect_ameli(src_ameli: Path, proc_imgs: Path, proc_masks: Path) -> list[str]:
    """
    Recorre src_ameli/ {train, validation, test}/images para cada fase:
      - Lee imagen.
      - Busca JSON (Annotation json format/<fase>/label) o TXT (Annotation txt format/<fase>/labels);
        si existe, genera máscara a partir de polígonos; si no, máscara vacía.
      - Binariza máscara, redimensiona imagen y máscara a 384×384.
      - Guarda con prefijo 'ameli_<fase>_' → produce nombres como 'ameli_train_0123.png'.
    Retorna lista de nombres guardados.
    """
    def json_to_mask(p: Path, shape):
        data = json.load(p.open("r", encoding="utf-8"))
        mask = np.zeros(shape, np.uint8)
        for shp in data.get("shapes", []):
            pts = np.array(shp["points"], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    def txt_to_mask(p: Path, shape):
        pts = []
        for line in p.read_text().splitlines():
            if "," in line:
                x, y = map(float, line.split(","))
                pts.append([int(x), int(y)])
        mask = np.zeros(shape, np.uint8)
        if pts:
            cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
        return mask

    am_files = []
    img_root  = src_ameli / "images"
    json_root = src_ameli / "Annotation json format"
    txt_root  = src_ameli / "Annotation txt format"

    for phase_dir in img_root.glob("*"):
        inner = phase_dir / "images"
        if not inner.exists():
            continue
        phase = phase_dir.name.lower()  # 'train', 'validation' o 'test'

        for img_path in inner.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]

            # Intentar JSON
            j = json_root / phase_dir.name / "label" / f"{img_path.stem}.json"
            t = txt_root  / phase_dir.name / "labels" / f"{img_path.stem}.txt"

            if j.exists():
                mask0 = json_to_mask(j, (H, W))
            elif t.exists():
                mask0 = txt_to_mask(t, (H, W))
            else:
                mask0 = np.zeros((H, W), np.uint8)

            # Binarizar máscara >127 → 255
            _, mask_bin = cv2.threshold(mask0, 127, 255, cv2.THRESH_BINARY)

            prefix = f"ameli_{phase}_"   # ameli_train_, ameli_validation_, ameli_test_
            stem   = img_path.stem      # 'IMG_0123'
            out_name = save_image_and_mask(
                img, mask_bin,
                proc_imgs, proc_masks,
                prefix, stem
            )
            am_files.append(out_name)

    print(f"[Ameli] procesadas {len(am_files)} imágenes (384×384, binarizadas).")
    return am_files

# ──────────────────────────────────────────────────────────────────────────────
# CREACIÓN DE CSVs DE SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def write_list_to_txt(path: Path, names: list[str]) -> None:
    path.write_text("\n".join(names), encoding="utf-8")

def create_splits_and_csvs(
    vt_files: list[str],
    am_files: list[str],
    out_dir: Path
) -> tuple[list[str], list[str], list[str]]:
    """
    - Mezcla aleatoria de vt_files con semilla.
    - Particiona vt_files en vt_train (70%), vt_val (15%), vt_test (15%).
    - train = vt_train + am_files.
    - Escribe en out_dir: train.csv, val.csv, test.csv.
    - Escribe: vt_train_10.csv, vt_train_25.csv, vt_train_50.csv (solo VT, sin Ameli).
    Devuelve (vt_train, vt_val, vt_test).
    """
    random.shuffle(vt_files)
    N = len(vt_files)
    n_train = int(0.70 * N)
    n_val   = int(0.15 * N)

    vt_train = vt_files[:n_train]
    vt_val   = vt_files[n_train : n_train + n_val]
    vt_test  = vt_files[n_train + n_val :]

    train_list = vt_train + am_files

    ensure_dir(out_dir)
    write_list_to_txt(out_dir / "train.csv", train_list)
    write_list_to_txt(out_dir / "val.csv",   vt_val)
    write_list_to_txt(out_dir / "test.csv",  vt_test)

    for pct in [0.10, 0.25, 0.50]:
        k = int(len(vt_train) * pct)
        write_list_to_txt(out_dir / f"vt_train_{int(pct*100)}.csv", vt_train[:k])

    print(f"[CSV] → train.csv ({len(train_list)}: VT_train {len(vt_train)} + Ameli {len(am_files)})")
    print(f"[CSV] → val.csv ({len(vt_val)}) | test.csv ({len(vt_test)})")
    print("[CSV] → vt_train_10, vt_train_25, vt_train_50 generados")
    return vt_train, vt_val, vt_test

# ──────────────────────────────────────────────────────────────────────────────
# CREACIÓN DE CARPETAS POR RÉGIMEN
# ──────────────────────────────────────────────────────────────────────────────

def create_regime_folders(
    vt_train: list[str],
    vt_val:   list[str],
    vt_test:  list[str],
    all_am_files: list[str],
    proc_imgs_dir: Path,
    proc_masks_dir: Path,
    regimes_root: Path
) -> None:
    """
    Para cada régimen (A-Full, Bxx, Cxx):
     - A-Full: copia vt_train (100%) con máscaras → train/images_labeled, masks_labeled
       deja images_unlabeled vacío.
     - Bxx: dentro de vt_train, toma xx% etiquetados → copia imágenes y máscaras
       a train/images_labeled, masks_labeled. Deja images_unlabeled vacío.
     - Cxx: dentro de vt_train, toma xx% etiquetados → copia imágenes y máscaras.
       Resto vt_train (100 - xx)% + all_am_files → copia SOLO imágenes a train/images_unlabeled.
     - Val/test para todos: vt_val/vt_test copiados completos (imágenes y máscaras).
    """
    # A-Full
    regime = "A-Full"
    base = regimes_root / regime

    # Train: preparar subcarpetas
    ensure_dir(base / "train" / "images_labeled")
    ensure_dir(base / "train" / "masks_labeled")
    ensure_dir(base / "train" / "images_unlabeled")  # vacío

    # Copiar 100% VT_train con máscaras
    for fname in vt_train:
        shutil.copy(proc_imgs_dir / fname, base / "train" / "images_labeled" / fname)
        shutil.copy(proc_masks_dir / fname, base / "train" / "masks_labeled" / fname)

    # Val/Test idénticos a VT_val / VT_test
    for split_name, split_list in [("val", vt_val), ("test", vt_test)]:
        ensure_dir(base / split_name / "images")
        ensure_dir(base / split_name / "masks")
        for fname in split_list:
            shutil.copy(proc_imgs_dir / fname, base / split_name / "images" / fname)
            shutil.copy(proc_masks_dir / fname, base / split_name / "masks" / fname)

    # —— Regímenes Bxx (Weak labels: no images_unlabeled en train)
    for pct in WEAK_PERCENTS:
        regime = f"B{pct}"
        base = regimes_root / regime

        # Crear estructura train/val/test
        ensure_dir(base / "train" / "images_labeled")
        ensure_dir(base / "train" / "masks_labeled")
        ensure_dir(base / "train" / "images_unlabeled")  # vacío

        ensure_dir(base / "val" / "images")
        ensure_dir(base / "val" / "masks")
        ensure_dir(base / "test" / "images")
        ensure_dir(base / "test" / "masks")

        # Dentro de vt_train, seleccionar pct% para etiquetados
        n_labeled = int((pct / 100.0) * len(vt_train))
        labeled_vt, _ = train_test_split(
            vt_train,
            train_size=n_labeled,
            random_state=SEED,
            shuffle=True
        )
        # Copiar etiquetados con máscaras
        for fname in labeled_vt:
            shutil.copy(proc_imgs_dir / fname, base / "train" / "images_labeled" / fname)
            shutil.copy(proc_masks_dir / fname, base / "train" / "masks_labeled" / fname)
        # images_unlabeled queda vacío

        # Val/Test idénticos a vt_val / vt_test
        for split_name, split_list in [("val", vt_val), ("test", vt_test)]:
            for fname in split_list:
                shutil.copy(proc_imgs_dir / fname, base / split_name / "images" / fname)
                shutil.copy(proc_masks_dir / fname, base / split_name / "masks" / fname)

    # —— Regímenes Cxx (Semi-supervised: vt_train_label + vt_train_no_label + Ameli en unlabeled)
    for pct in SEMI_PERCENTS:
        regime = f"C{pct}"
        base = regimes_root / regime

        # Crear estructura train/val/test
        ensure_dir(base / "train" / "images_labeled")
        ensure_dir(base / "train" / "masks_labeled")
        ensure_dir(base / "train" / "images_unlabeled")

        ensure_dir(base / "val" / "images")
        ensure_dir(base / "val" / "masks")
        ensure_dir(base / "test" / "images")
        ensure_dir(base / "test" / "masks")

        # Dentro de vt_train, seleccionar pct% etiquetados, el resto no etiquetados
        n_labeled = int((pct / 100.0) * len(vt_train))
        labeled_vt, rem_vt = train_test_split(
            vt_train,
            train_size=n_labeled,
            random_state=SEED,
            shuffle=True
        )

        # (1) Copiar etiquetados → images_labeled + masks_labeled
        for fname in labeled_vt:
            shutil.copy(proc_imgs_dir / fname, base / "train" / "images_labeled" / fname)
            shutil.copy(proc_masks_dir / fname, base / "train" / "masks_labeled" / fname)

        # (2) Copiar VT restante (rem_vt) solo imágenes → images_unlabeled
        for fname in rem_vt:
            shutil.copy(proc_imgs_dir / fname, base / "train" / "images_unlabeled" / fname)

        # (3) Copiar TODAS las imágenes Ameli → images_unlabeled
        for fname in all_am_files:
            shutil.copy(proc_imgs_dir / fname, base / "train" / "images_unlabeled" / fname)

        # Val/Test idénticos a VT_val / VT_test
        for split_name, split_list in [("val", vt_val), ("test", vt_test)]:
            for fname in split_list:
                shutil.copy(proc_imgs_dir / fname, base / split_name / "images" / fname)
                shutil.copy(proc_masks_dir / fname, base / split_name / "masks" / fname)

    print(f"✅ Carpetas por régimen creadas en: {regimes_root}")

# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Preprocesar VT+Ameli y crear carpetas por régimen"
    )
    ap.add_argument(
        "--src_vt", type=Path, required=False,
        help="Carpeta raíz VT 512×512 (busca recursivamente subcarpetas 'images_512')."
    )
    ap.add_argument(
        "--src_ameli", type=Path, required=False,
        help="Carpeta raíz Ameli (busca 'images/<FASE>/images/*.png')."
    )
    ap.add_argument(
        "--out", type=Path, required=False,
        help="Carpeta raíz de salida (se creará '<out>/processed' y '<out>/regimes')."
    )
    args = ap.parse_args()

    # Rutas por defecto si no se pasan por línea de comandos
    src_vt    = args.src_vt    or VT_ROOT_DEFAULT
    src_ameli = args.src_ameli or AMELI_ROOT_DEFAULT
    out_root  = args.out       or OUT_ROOT_DEFAULT

    # Directorios internos
    proc_imgs = out_root / "processed" / "images"
    proc_masks= out_root / "processed" / "masks"
    csv_dir   = out_root / "processed"
    regimes_dir = out_root / "regimes"

    # 1) Crear directorios base
    ensure_dir(proc_imgs)
    ensure_dir(proc_masks)
    ensure_dir(csv_dir)
    ensure_dir(regimes_dir)

    # 2) Recolectar y procesar VT (resize 384×384 + binarizar máscaras + prefijo 'vt_<subset>_')
    vt_files = collect_vt(src_vt, proc_imgs, proc_masks)
    if not vt_files:
        raise RuntimeError(f"No se encontraron imágenes VT en {src_vt}")

    # 3) Recolectar y procesar Ameli (resize 384×384, binarizar, prefijo 'ameli_<phase>_')
    am_files = collect_ameli(src_ameli, proc_imgs, proc_masks)
    # Puede dejarse am_files vacío si no hay anotaciones; en ese caso, Cxx tendrá solo VT sin etiquetar.

    # 4) Crear CSVs de splits y obtener listas vt_train / vt_val / vt_test
    vt_train, vt_val, vt_test = create_splits_and_csvs(vt_files, am_files, csv_dir)

    # 5) Crear carpetas por régimen copiando imágenes/máscaras desde 'processed'
    create_regime_folders(
        vt_train, vt_val, vt_test,
        am_files,
        proc_imgs, proc_masks,
        regimes_dir
    )

    print("¡Listo! Todo el flujo ha finalizado.")

if __name__ == "__main__":
    main()
