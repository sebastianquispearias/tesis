import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp

# ----------------------------------------------------
# 0) CONFIGURACIÓN GLOBAL Y SEED
# ----------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "data/all_results/regimes" # Asegúrate que esta sea tu ruta base
BACKBONE = 'efficientnet-b3'
INPUT_SHAPE = (384, 384)
LR = 1e-4
BATCH_SIZE = 2 # Ajusta según tu VRAM (con 12GB, 2 o 4 debería funcionar)
EPOCHS = 80
N_CLASSES = 1 # Segmentación binaria

# ----------------------------------------------------
# A) DATASET Y DATALOADERS
# ----------------------------------------------------
class CorrosionDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_fps = sorted([os.path.join(images_dir, fname) for fname in os.listdir(images_dir)])
        self.masks_fps = sorted([os.path.join(masks_dir, fname) for fname in os.listdir(masks_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.images_fps)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.images_fps[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype('float32')

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        
        return img, mask.unsqueeze(0) # Añadir dimensión de canal a la máscara

# Augmentations (consistente con tu otro script)

train_transform = A.Compose([
    A.Resize(*INPUT_SHAPE),
    # Geométricas
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5), # Añadido para consistencia
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    # Fotométricas
    A.RandomBrightnessContrast(p=0.5),
    # Enmascarado
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    # Normalización y conversión a Tensor
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(*INPUT_SHAPE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ----------------------------------------------------
# B) PARSEO DE ARGUMENTOS
# ----------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento supervisado para baselines Bxx")
    p.add_argument("--regime", required=True,
                   choices=["B10", "B25", "B50", "B75", "A-Full"],
                   help="Régimen de datos a utilizar.")
    return p.parse_args()

# ----------------------------------------------------
# C) BUCLE DE ENTRENAMIENTO Y VALIDACIÓN
# ----------------------------------------------------
def main():
    args = parse_args()
    regime = args.regime
    
    # Configurar logging
    log_file = f"entrenamiento_log_{regime}_SUPERVISADO.txt"
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
                        format="%(asctime)s %(levelname)s: %(message)s")
    
    logging.info(f"--- Iniciando Entrenamiento Supervisado para el Régimen: {regime} ---")
    logging.info(f"Usando dispositivo: {DEVICE}")

    # Rutas
    train_dir = os.path.join(DATA_ROOT, regime, "train")
    val_dir = os.path.join(DATA_ROOT, regime, "val")
    
    # DataLoaders
    train_ds = CorrosionDataset(os.path.join(train_dir, "images_labeled"), os.path.join(train_dir, "masks_labeled"), transform=train_transform)
    val_ds = CorrosionDataset(os.path.join(val_dir, "images"), os.path.join(val_dir, "masks"), transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Lógica de congelamiento del encoder
    freeze_encoder = regime.startswith("B")
    logging.info(f"Congelar encoder (freeze_encoder): {freeze_encoder}")

    # Modelo
    model = smp.DeepLabV3Plus(
        encoder_name=BACKBONE,
        encoder_weights="imagenet",
        in_channels=3,
        classes=N_CLASSES,
        encoder_freeze=freeze_encoder # <-- Lógica de congelamiento aplicada aquí
    ).to(DEVICE)
    
    # Loss y Optimizador
    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Métricas
    metrics = {
        "iou": smp.metrics.iou_score,
        "f1": smp.metrics.f1_score
    }

    best_iou = 0.0
    model_save_path = f"best_model_supervised_{regime}.pth"

    # Bucle Principal
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loop = tqdm(train_loader, total=len(train_loader), leave=False)
        epoch_loss = 0.0
        
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Forward
            pred = model(x)
            loss = loss_fn(pred, y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        # Validación
        model.eval()
        val_iou, val_f1, val_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
                pred_val = model(x_val)
                val_loss += loss_fn(pred_val, y_val).item()
                # Para métricas, las predicciones deben ser probabilidades y las máscaras enteros
                probs = torch.sigmoid(pred_val)
                val_iou += metrics["iou"](probs, y_val.long()).item()
                val_f1 += metrics["f1"](probs, y_val.long()).item()

        # Promediar métricas de validación
        val_iou /= len(val_loader)
        val_f1 /= len(val_loader)
        val_loss /= len(val_loader)

        logging.info(f"Epoch {epoch}: Train Loss={epoch_loss/len(train_loader):.4f} | Val Loss={val_loss:.4f} | Val IoU={val_iou:.4f} | Val F1={val_f1:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"--> Nuevo mejor modelo guardado en {model_save_path} con Val IoU: {best_iou:.4f}")

    logging.info(f"Entrenamiento completado. Mejor Val IoU: {best_iou:.4f}")
    # Aquí iría la evaluación final sobre el conjunto de test, cargando el mejor modelo guardado.

if __name__ == '__main__':
    main()