# viz_augmentations.py

import os
import cv2
import matplotlib.pyplot as plt
import albumentations as A

# 1) Ajusta esta ruta al directorio donde tienes tus imágenes de train
DATA_DIR = r'C:\Users\User\Desktop\tesis\data\320x320'
x_train_dir = os.path.join(DATA_DIR, 'Train', 'images')

# 2) Lee una imagen de ejemplo
lista_imgs = os.listdir(x_train_dir)
if not lista_imgs:
    raise ValueError("No hay imágenes en " + x_train_dir)
img_path = os.path.join(x_train_dir, lista_imgs[0])
image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

# 3) Define aquí las mismas augmentations que usas en tu código:
INPUT_SHAPE = (384, 384)
transforms = {
    "Original":           A.Compose([A.Resize(*INPUT_SHAPE)]),
    "HorizontalFlip":     A.Compose([A.Resize(*INPUT_SHAPE), A.HorizontalFlip(p=1.0)]),
    "VerticalFlip":       A.Compose([A.Resize(*INPUT_SHAPE), A.VerticalFlip(p=1.0)]),
    "RandomRotate90":     A.Compose([A.Resize(*INPUT_SHAPE), A.RandomRotate90(p=1.0)]),
    "Transpose":          A.Compose([A.Resize(*INPUT_SHAPE), A.Transpose(p=1.0)]),
    "GridDistortion":     A.Compose([A.Resize(*INPUT_SHAPE), A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0)]),
}

# 4) Aplica y dibuja en un grid 2×3
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for ax, (name, aug) in zip(axes, transforms.items()):
    aug_img = aug(image=image)['image']
    ax.imshow(aug_img)
    ax.set_title(name)
    ax.axis('off')

plt.tight_layout()
plt.show()
