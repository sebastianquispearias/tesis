import os
import sys
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import numpy as np

# ----------------------------------------------------
# 0) CONFIGURACIÓN GLOBAL
# ----------------------------------------------------
# Estos parámetros deben ser los mismos que se usaron durante el entrenamiento
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = 'efficientnet-b3'
INPUT_SHAPE = (384, 384)
BATCH_SIZE = 2 # Puede ser más grande para evaluación si tu VRAM lo permite
N_CLASSES = 1

# ----------------------------------------------------
# A) DATASET Y TRANSFORMACIONES (Copiado de tu script de entrenamiento)
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
        
        return img, mask.unsqueeze(0)

# Solo necesitamos la transformación de validación/test
val_transform = A.Compose([
    A.Resize(*INPUT_SHAPE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ----------------------------------------------------
# B) PARSEO DE ARGUMENTOS
# ----------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Script de Evaluación de Modelos de Segmentación")
    p.add_argument("--weights", required=True, help="Ruta al archivo de pesos del modelo (.pth)")
    p.add_argument("--regime", required=True, help="Régimen de datos a usar para el test (ej: B25, C50)")
    p.add_argument("--data_root", default="data/all_results/regimes", help="Ruta base a los directorios de regímenes")
    return p.parse_args()

# ----------------------------------------------------
# C) FUNCIÓN PRINCIPAL DE EVALUACIÓN
# ----------------------------------------------------
def main():
    args = parse_args()
    print(f"--- Iniciando Evaluación del Modelo ---")
    print(f"  - Pesos: {args.weights}")
    print(f"  - Conjunto de Test del Régimen: {args.regime}")
    print(f"  - Usando dispositivo: {DEVICE}")

    # 1. Construir el Modelo (debe ser idéntico al que se entrenó)
    model = smp.DeepLabV3Plus(
        encoder_name=BACKBONE,
        encoder_weights=None,  # No necesitamos cargar pesos de ImageNet, cargaremos los nuestros
        in_channels=3,
        classes=N_CLASSES
    ).to(DEVICE)

    # 2. Cargar los pesos entrenados
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    print("--> Pesos cargados exitosamente.")
    
    # 3. Preparar el DataLoader de Test
    test_dir = os.path.join(args.data_root, args.regime, "test")
    test_ds = CorrosionDataset(os.path.join(test_dir, "images"), os.path.join(test_dir, "masks"), transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 4. Definir Loss y Métricas
    loss_fn = smp.losses.DiceLoss(mode='binary')
    
    # 5. Bucle de Evaluación
    model.eval() # Poner el modelo en modo de evaluación
    test_iou, test_f1, test_loss = 0.0, 0.0, 0.0
    with torch.no_grad(): # No necesitamos calcular gradientes
        loop = tqdm(test_loader, total=len(test_loader), desc="Evaluando en Test")
        for x_test, y_test in loop:
            x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)
            
            pred_test = model(x_test)
            test_loss += loss_fn(pred_test, y_test).item()
            
            probs = torch.sigmoid(pred_test)
            y_test_int = y_test.long()
            
            tp, fp, fn, tn = smp.metrics.get_stats(probs, y_test_int, mode='binary', threshold=0.5)
            test_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro').item()
            test_f1 += smp.metrics.f1_score(tp, fp, fn, tn, reduction='macro').item()
    
    # 6. Calcular y mostrar los resultados finales
    num_batches = len(test_loader)
    final_loss = test_loss / num_batches
    final_iou = test_iou / num_batches
    final_f1 = test_f1 / num_batches
    
    print("\n--- ¡Evaluación Completada! ---")
    print(f">> RESULTADO FINAL EN TEST (Régimen: {args.regime}):")
    print(f"   - Loss: {final_loss:.4f}")
    print(f"   - IoU:  {final_iou:.4f}")
    print(f"   - F1:   {final_f1:.4f}")

if __name__ == '__main__':
    main()