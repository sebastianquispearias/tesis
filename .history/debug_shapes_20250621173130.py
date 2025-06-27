from main_semisup_tesis import MixedDataLoader, build_model, get_augmentations
from metrics_utils import DiceLoss, get_consistency_weight
import torch

# --- Parámetros mínimos para arrancar un loader ---
batch_size = 2
arch = 'deeplabv3+'

# Asume que tu MixedDataLoader se construye así:
# loader = MixedDataLoader(labeled_dataset, unlabeled_dataset, batch_size, augmentations...)
loader = MixedDataLoader(
    train_labeled_images='train/images_labeled',
    train_labeled_masks='train/masks_labeled',
    train_unlabeled_images='train/images_unlabeled',
    batch_size=batch_size,
    strong_aug=get_augmentations(strong=True),
    weak_aug=get_augmentations(strong=False),
)

x_lab, y_lab, x_unl_stu, x_unl_tea = next(iter(loader))
print("x_lab:", tuple(x_lab.shape))
print("y_lab:", tuple(y_lab.shape))
print("x_unl_student:", tuple(x_unl_stu.shape))
print("x_unl_teacher:", tuple(x_unl_tea.shape))
