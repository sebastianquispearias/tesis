import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1) Carga modelo y pesos
model = build_model()  # o como lo definas en models.py
model.load_weights('checkpoints/best_C75.h5')

# 2) Carga una imagen de test y su máscara
img = cv2.imread('data/all_results/regimes/C75/test/images/vt_test_6.png')
gt = cv2.imread('data/all_results/regimes/C75/test/masks/vt_test_6.png', 0) > 0

# 3) Preprocesa y predice
inp = preprocess(img)                # según tu pipeline
p = model.predict(np.expand_dims(inp,0))[0,...,0]
pred_mask = p > 0.5                  # tú eliges el umbral

# 4) Calcula IoU  
intersection = np.logical_and(gt, pred_mask).sum()
union        = np.logical_or (gt, pred_mask).sum()
iou          = intersection / union
print(f"IoU de esta inferencia: {iou:.3f}")

# 5) Visualización
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(img[...,::-1]); plt.title('Imagen')
plt.subplot(1,3,2); plt.imshow(gt, cmap='gray');   plt.title('Ground Truth')
plt.subplot(1,3,3); plt.imshow(pred_mask, cmap='gray'); plt.title('Predicción')
plt.show()
