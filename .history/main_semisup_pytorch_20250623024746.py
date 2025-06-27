# main_semisup_pytorch.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import random
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import crear_kernel_elipse, actualizar_teacher_ema
from losses import LossConsistenciaMorfologicaCompuesta
# ─────────── 0) Parámetros globales ───────────
SEED          = 42
BATCH_SIZE    = 2
LR            = 1e-4
EPOCHS        = 80
INPUT_SHAPE   = (384, 384)
DATA_ROOT     = r"./data/all_results/regimes"  # ajusta si hace falta
CLASSES       = ["corrosion"]
N_CLASSES     = 1
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# Mean Teacher
EMA_ALPHA     = 0.99
CONS_MAX      = 1.0
CONS_RAMPUP   = 30
UNLABELED_W   = 0.1

def get_consistency_weight(epoch):
    if epoch >= CONS_RAMPUP:
        return CONS_MAX
    phase = 1.0 - epoch/CONS_RAMPUP
    return CONS_MAX * np.exp(-5 * phase * phase)  # :contentReference[oaicite:4]{index=4}

# ─────────── 1) Argumentos ───────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--regime", required=True,
                   choices=["C10","C25","C50","C75"],
                   help="Carpeta de régimen dentro de data/all_results/regimes")
    return p.parse_args()

# ─────────── 2) Dataset ───────────
class CorrosionDataset(Dataset):
    def __init__(self, images_dir, masks_dir, 
                 transform=None, mode="both"):
        self.images = sorted(os.listdir(images_dir))
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.transform  = transform
        self.mode       = mode  # "labeled", "unlabeled", "both"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        img = np.array(Image.open(os.path.join(self.images_dir, fname)).convert("RGB"))
        if self.mode in ("both","labeled"):
            mask = np.array(Image.open(os.path.join(self.masks_dir, fname)).convert("L"))
            mask = (mask>0).astype("float32")[...,None]
        else:
            mask = np.zeros((img.shape[0],img.shape[1],1),dtype="float32")

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
            # --- aquí convertimos mask a [1, H, W] tensor ---
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).permute(2,0,1).float()
            else:
                # albumentations a veces devuelve Tensor con shape [H,W,1]
                if mask.ndim==3 and mask.shape[2]==1:
                    mask = mask.permute(2,0,1)
                elif mask.ndim==2:
                    mask = mask.unsqueeze(0)
        else:
            img = ToTensorV2()(image=img)["image"]
            mask = torch.from_numpy(mask).permute(2,0,1).float()

        return img, mask

# ─────────── 3) Transforms ───────────
# ─────────── 3) Transforms ───────────
train_strong = A.Compose([
    A.Resize(*INPUT_SHAPE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

train_weak = A.Compose([
    A.Resize(*INPUT_SHAPE),
    A.HorizontalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.1),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(*INPUT_SHAPE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])


# ─────────── 4) Main ───────────
def main():
    args = parse_args()
    regime = args.regime
    # Logging a archivo
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler("entrenamiento_log_pytorch.txt"),
                                  logging.StreamHandler(sys.stdout)],
                        format="%(asctime)s %(levelname)s: %(message)s")
    logging.info(f"Dispositivo: {DEVICE}")

    # Rutas
    root = os.path.join(DATA_ROOT, regime)
    xl = os.path.join(root, "train/images_labeled")
    yl = os.path.join(root, "train/masks_labeled")
    xu = os.path.join(root, "train/images_unlabeled")
    xv = os.path.join(root, "val/images")
    yv = os.path.join(root, "val/masks")

    # DataLoaders
    # DataLoaders
    sup_ds = CorrosionDataset(xl, yl, transform=train_strong, mode="labeled")
    sup_loader = DataLoader(sup_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

    unlab_strong_ds = CorrosionDataset(xu, xu, transform=train_strong, mode="unlabeled")
    unlab_strong_loader = DataLoader(unlab_strong_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

    unlab_weak_ds = CorrosionDataset(xu, xu, transform=train_weak, mode="unlabeled")
    unlab_weak_loader = DataLoader(unlab_weak_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

    val_ds = CorrosionDataset(xv, yv, transform=val_transform, mode="labeled")
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Modelos
    student = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=N_CLASSES,
        activation=None,
    ).to(DEVICE)

    teacher = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=N_CLASSES,
        activation=None,
    ).to(DEVICE)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters(): p.requires_grad = False

    # Optimizador y pérdidas
    optimizer = optim.Adam(student.parameters(), lr=LR)
    sup_loss   = smp.losses.DiceLoss(mode='binary')
    # Instancia tu loss morfológico compuesto
    cons_loss  = torch.nn.MSELoss() 
    logging.info("Usando Loss de Consistencia Clásico (MSE) para el baseline.")

    best_iou = 0.0
    for epoch in range(1, EPOCHS+1):
        student.train()
        epoch_sup, epoch_cons = 0.0, 0.0
        loop = tqdm(zip(sup_loader, unlab_strong_loader, unlab_weak_loader), total=len(sup_loader), leave=False)

        for (x_s, y_s), (x_us, _), (x_uw, _) in loop:
            x_s, y_s = x_s.to(DEVICE), y_s.to(DEVICE)
            x_us = x_us.to(DEVICE)   # student on strong
            x_uw = x_uw.to(DEVICE)   # teacher on weak

            # Forward supervised
            log_s = student(x_s)
            loss_s = sup_loss(log_s, y_s)

            # Forward unsupervised
            with torch.no_grad():
                t_uw = torch.sigmoid(teacher(x_uw))
            s_us = torch.sigmoid(student(x_us))

            w = get_consistency_weight(epoch)
            loss_c = cons_loss(s_us, t_uw) * w * UNLABELED_W

            loss = loss_s + loss_c
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update...

            # EMA update
            for ps, pt in zip(student.parameters(), teacher.parameters()):
                pt.data.mul_(EMA_ALPHA).add_(ps.data * (1-EMA_ALPHA))

            epoch_sup  += loss_s.item()
            epoch_cons += loss_c.item()
            loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
            loop.set_postfix(loss_sup=loss_s.item(), loss_cons=loss_c.item())
        # ——— Validación (pérdida + métricas) ———
        student.eval()
        epoch_val_loss = 0.0
        iou, prec, rec, f1 = 0.0, 0.0, 0.0, 0.0
        n_val = 0
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
            with torch.no_grad():
                logits_val = student(x_val)
                # val_loss sobre BCEWithLogits
                loss_val = sup_loss(logits_val, y_val)
                epoch_val_loss += loss_val.item()
                pred = torch.sigmoid(logits_val) > 0.5
            # métricas por batch
            tp =   (pred & (y_val>0.5)).sum().item()
            fp =   (pred & ~(y_val>0.5)).sum().item()
            fn =   (~pred & (y_val>0.5)).sum().item()
            union = tp + fp + fn
            iou  += tp/union        if union>0   else 0
            prec += tp/(tp+fp)      if tp+fp>0   else 0
            rec  += tp/(tp+fn)      if tp+fn>0   else 0
            f1   += 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0
            n_val += 1

        # medias
        val_loss = epoch_val_loss / len(val_loader)
        iou, prec, rec, f1 = [x/n_val for x in (iou,prec,rec,f1)]
        sup_avg  = epoch_sup  / len(sup_loader)
        cons_avg = epoch_cons / len(sup_loader)
        total    = sup_avg + cons_avg
        logging.info(
            f"Época {epoch}: "
            f"sup={sup_avg:.4f}, cons={cons_avg:.4f}, total={total:.4f} | "
            f"val_loss={val_loss:.4f}, IoU={iou:.4f}, P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}"
        )
        # Guarda mejor modelo por IoU
        if iou > best_iou:
            best_iou = iou
            torch.save(student.state_dict(), f"best_deeplab_{regime}.pth")
    # ——— Evaluación final en TEST ———
    # Rutas de test
    xt = os.path.join(root, "test/images")
    yt = os.path.join(root, "test/masks")
    test_ds    = CorrosionDataset(xt, yt, transform=val_transform, mode="labeled")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Carga el mejor modelo
    student.load_state_dict(torch.load(f"best_deeplab_{regime}.pth"))
    student.eval()

    # Cálculo de test_loss y test_iou
    test_loss, test_iou = 0.0, 0.0
    n_test = 0
    for x_t, y_t in test_loader:
        x_t, y_t = x_t.to(DEVICE), y_t.to(DEVICE)
        with torch.no_grad():
            logits_t = student(x_t)
            loss_t = sup_loss(logits_t, y_t)
            test_loss += loss_t.item()
            pred_t = torch.sigmoid(logits_t) > 0.5
        tp = (pred_t & (y_t>0.5)).sum().item()
        union = (pred_t | (y_t>0.5)).sum().item()
        test_iou += tp/union if union>0 else 0
        n_test += 1
    test_loss /= n_test
    test_iou  /= n_test
    logging.info(f"Test final: Loss={test_loss:.4f}, IoU={test_iou:.4f}")
if __name__ == "__main__":
    # reproducibilidad
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE=="cuda": torch.cuda.manual_seed(SEED)
    main()
