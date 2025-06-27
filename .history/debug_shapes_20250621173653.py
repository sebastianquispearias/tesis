# debug_shapes.py
from main_semisup_tesis import (
    MixedDataLoader,
    get_training_augmentation,
    get_validation_augmentation,
    get_preprocessing,
    preprocess_input
)
import numpy as np

# Ajusta estas rutas a tu estructura
x_lab_dir   = r"C:\Users\User\Desktop\tesis\data\all_results\regimes\C25\train\images_labeled"
y_lab_dir   = r"C:\Users\User\Desktop\tesis\data\all_results\regimes\C25\train\masks_labeled"
x_unlab_dir = r"C:\Users\User\Desktop\tesis\data\all_results\regimes\C25\train\images_unlabeled"

batch_size = 2

# Instanciar MixedDataLoader con los argumentos correctos
loader = MixedDataLoader(
    x_lab_dir, y_lab_dir, x_unlab_dir,
    batch_size_lab       = batch_size,
    batch_size_unlab     = batch_size,
    augment_lab          = get_training_augmentation(),
    augment_unlab_student= get_training_augmentation(),
    augment_unlab_teacher= get_validation_augmentation(),
    preprocess           = get_preprocessing(preprocess_input)
)

# Saca un solo lote y muestra shapes
x_lab, y_lab, x_unl_s, x_unl_t = next(iter(loader))
print("x_lab shape:          ", x_lab.shape)
print("y_lab shape:          ", y_lab.shape)
print("x_unl_student shape:  ", x_unl_s.shape)
print("x_unl_teacher shape:  ", x_unl_t.shape)
