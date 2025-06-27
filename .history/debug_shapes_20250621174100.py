# debug_shapes.py
import sys
# 1) Importamos tu módulo (que *redirige* stdout/stderr)
import main_semisup_tesis as m

# 2) Restauramos la salida a consola
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

from main_semisup_tesis import (
    MixedDataLoader,
    get_training_augmentation,
    get_validation_augmentation,
    get_preprocessing,
    preprocess_input
)
import numpy as np

x_lab_dir   = r"C:\Users\User\Desktop\tesis\data\all_results\regimes\C25\train\images_labeled"
y_lab_dir   = r"C:\Users\User\Desktop\tesis\data\all_results\regimes\C25\train\masks_labeled"
x_unlab_dir = r"C:\Users\User\Desktop\tesis\data\all_results\regimes\C25\train\images_unlabeled"

batch_size = 2

loader = MixedDataLoader(
    x_lab_dir, y_lab_dir, x_unlab_dir,
    batch_size_lab       = batch_size,
    batch_size_unlab     = batch_size,
    augment_lab          = get_training_augmentation(),
    augment_unlab_student= get_training_augmentation(),
    augment_unlab_teacher= get_validation_augmentation(),
    preprocess           = get_preprocessing(preprocess_input)
)

x_lab, y_lab, x_unl_s, x_unl_t = next(iter(loader))
print("x_lab shape:         ", x_lab.shape)
print("y_lab shape:         ", y_lab.shape)
print("x_unl_student shape: ", x_unl_s.shape)
print("x_unl_teacher shape: ", x_unl_t.shape)
