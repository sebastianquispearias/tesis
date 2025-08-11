import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Para evitar el error de OpenMP en Windows

import argparse
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# --- PARÁMETROS GLOBALES (deben coincidir con el entrenamiento) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = 'efficientnet-b3'
INPUT_SHAPE = (384, 384)
N_CLASSES = 1

# --- TRANSFORMACIÓN PARA PREPROCESAR LA IMAGEN ---
preprocess_transform = A.Compose([
    A.Resize(*INPUT_SHAPE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def parse_args():
    p = argparse.ArgumentParser(description='Inferencia de Segmentación de Corrosión con PyTorch')
    p.add_argument('--weights', required=True, help='Ruta al archivo de pesos del modelo (.pth)')
    p.add_argument('--input', required=True, help='Ruta a la imagen de entrada para inferencia')
    p.add_argument('--mask', help='(Opcional) Ruta a la máscara de ground truth para calcular IoU')
    p.add_argument('--threshold', type=float, default=0.5, help='Umbral para binarizar la predicción')
    p.add_argument('--no-show', action='store_true', help='Opción para no mostrar la visualización de Matplotlib')
    return p.parse_args()


def main():
    args = parse_args()

    # --- 1. CONSTRUIR Y CARGAR MODELO ---
    print(f"INFO: Usando dispositivo: {DEVICE}")
    print(f"INFO: Construyendo modelo DeepLabV3+ con backbone {BACKBONE}...")
    
    model = smp.DeepLabV3Plus(
        encoder_name=BACKBONE,
        encoder_weights=None, # Los pesos se cargan desde el archivo, no de ImageNet
        in_channels=3,
        classes=N_CLASSES
    ).to(DEVICE)

    # Cargar los pesos entrenados
    try:
        model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
        print(f"INFO: Pesos cargados exitosamente desde: {args.weights}")
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de pesos en: {args.weights}")
        return
        
    model.eval() # Poner el modelo en modo de evaluación

    # --- 2. CARGAR Y PREPROCESAR IMAGEN DE ENTRADA ---
    try:
        img_bgr = cv2.imread(args.input)
        if img_bgr is None: raise FileNotFoundError
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except FileNotFoundError:
        print(f"ERROR: No se encontró la imagen de entrada en: {args.input}")
        return

    # Aplicar transformaciones
    transformed = preprocess_transform(image=img_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(DEVICE) # Añadir dimensión de batch

    # --- 3. REALIZAR INFERENCIA ---
    print("INFO: Realizando inferencia...")
    with torch.no_grad():
        pred_logits = model(input_tensor)
        pred_probs = torch.sigmoid(pred_logits)

    # Procesar la salida: quitar dimensión de batch, mover a CPU, binarizar
    pred_mask = (pred_probs.squeeze().cpu().numpy() > args.threshold).astype(np.uint8)
    print(f"INFO: Predicción realizada con umbral de {args.threshold}")

    # --- 4. CALCULAR IoU (SI SE PROPORCIONA MÁSCARA) ---
    iou = None
    if args.mask:
        try:
            gt_bgr = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
            if gt_bgr is None: raise FileNotFoundError
            # Redimensionar ground truth para que coincida con el tamaño de la predicción
            gt_resized = cv2.resize(gt_bgr, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt_mask = (gt_resized > 0)
            
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            iou = intersection / union if union > 0 else 0.0
            print(f"INFO: IoU calculado: {iou:.4f}")
        except FileNotFoundError:
            print(f"ADVERTENCIA: No se encontró la máscara en: {args.mask}. No se calculará el IoU.")

    # --- 5. VISUALIZACIÓN ---
    if not args.no_show:
        print("INFO: Mostrando resultados...")
        
        # Redimensionar la predicción al tamaño original de la imagen para una mejor visualización
        pred_mask_original_size = cv2.resize(pred_mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        num_plots = 1
        if args.mask: num_plots = 3
        
        plt.figure(figsize=(15, 5))

        # Plot 1: Imagen Original
        plt.subplot(1, num_plots, 1)
        plt.imshow(img_rgb)
        plt.title('Imagen de Entrada')
        plt.axis('off')

        if args.mask:
            # Plot 2: Ground Truth
            plt.subplot(1, num_plots, 2)
            plt.imshow(gt_mask, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')

            # Plot 3: Predicción
            plt.subplot(1, num_plots, 3)
            plt.imshow(pred_mask_original_size, cmap='gray')
            plt.title(f'Predicción (IoU: {iou:.4f})' if iou is not None else 'Predicción')
            plt.axis('off')
        else:
            # Si no hay máscara, mostrar la predicción en el segundo subplot
            plt.subplot(1, 2, 2)
            plt.imshow(pred_mask_original_size, cmap='gray')
            plt.title('Predicción')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()


#python inferencia_pytorch.py --weights best_deeplab_C50_ENHANCER.pth --input "C:\Users\User\Desktop\tesis\data\all_results\regimes\A-Full\test\images\vt_test_6.png" --mask "C:\Users\User\Desktop\tesis\data\all_results\regimes\A-Full\test\masks\vt_test_6.png"
    # best_deeplab_C25_ENHANCER.pth
    #python inferencia_pytorch.py --weights best_model_supervised_B50.pth --input "C:\Users\User\Desktop\tesis\data\all_results\regimes\A-Full\test\images\vt_test_6.png" --mask "C:\Users\User\Desktop\tesis\data\all_results\regimes\A-Full\test\masks\vt_test_6.png"
    #python inferencia_pytorch.py --weights best_deeplab_C25.pth --input "C:\Users\User\Desktop\tesis\data\all_results\regimes\A-Full\test\images\vt_test_6.png" --mask "C:\Users\User\Desktop\tesis\data\all_results\regimes\A-Full\test\masks\vt_test_6.png"
 #best_deeplab_C50_1KERNEL.pth

#python inferencia_pytorch.py --weights best_deeplab_D50_1KERNEL.pth --input "C:\Users\User\Desktop\tesis\data\all_results\regimes\A-Full\test\images\vt_test_6.png" --mask "C:\Users\User\Desktop\tesis\data\all_results\regimes\A-Full\test\masks\vt_test_6.png"
#best_model_supervised_B50.pth

#python inferencia_pytorch.py --weights best_model_supervised_B50.pth --input "C:\Users\User\Desktop\tesis\data\all_results\regimes\A-Full\test\images\vt_test_6.png" --mask "C:\Users\User\Desktop\tesis\data\all_results\regimes\A-Full\test\masks\vt_test_6.png"
#best_deeplab_C75_WANG.pth
#python inferencia_pytorch.py --weights best_model_supervised_B50.pth --input "C:\Users\User\Desktop\tesis\data\all_results\regimes\A-Full\test\images\vt_test_6.png" --mask "C:\Users\User\Desktop\tesis\data\all_results\regimes\A-Full\test\masks\vt_test_6.png"
#python inferencia_pytorch.py --weights best_deeplab_D75_1kernel.pth --input "C:\Users\User\Desktop\tesis\data\all_results\regimes\A-Full\test\images\vt_train_253.png" --mask "C:\Users\User\Desktop\tesis\data\all_results\regimes\A-Full\test\masks\vt_train_253.png"
#python inferencia_pytorch.py --weights best_deeplab_C75w.01_wang.pth --input "C:\Users\User\Desktop\tesis\data\all_results\regimes\A-Full\test\images\vt_train_253.png" --mask "C:\Users\User\Desktop\tesis\data\all_results\regimes\A-Full\test\masks\vt_train_253.png"
