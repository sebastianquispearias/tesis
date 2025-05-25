# main.py
import os
#0 = DEBUG, 1 = INFO, 2 = WARNING, 3 = ERROR (por defecto)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_XLA_FLAGS']  = '--xla_hlo_profile'  # opcional, para XLA

import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models as sm
import cv2
from tensorflow import keras
import tensorflow as tf
from callbacks_monitor import ProgressMonitor

tf.config.run_functions_eagerly(True)


from models import build_model

import albumentations as A





# --- Configuración de logging ---
log = open('entrenamiento_log.txt', 'w', encoding='utf-8')
sys.stdout = log
sys.stderr = log

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(log)]
)

# --- Parámetros globales ---
DATA_DIR = r'C:\Users\User\Desktop\tesis\data\320x320'
MODEL_NAME = 'Unet_EfficientnetB3_final'
BACKBONE = 'efficientnetb3'
BATCH_SIZE = 1
CLASSES = ['corrosion']
LR = 1e-4
EPOCHS = 1
print(EPOCHS)
INPUT_SHAPE = (384, 384, 3)

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)
activation = 'sigmoid' if n_classes == 1 else 'softmax'

# Directorios de datos
x_train_dir = os.path.join(DATA_DIR, 'Train', 'images')
y_train_dir = os.path.join(DATA_DIR, 'Train', 'masks')

x_valid_dir = os.path.join(DATA_DIR, 'Val', 'images')
y_valid_dir = os.path.join(DATA_DIR, 'Val', 'masks')

x_test_dir = os.path.join(DATA_DIR, 'Test', 'images')
y_test_dir = os.path.join(DATA_DIR, 'Test', 'masks')

# --- Data augmentation (solo para train) ---
def get_training_augmentation():
    return A.Compose([
        # mismo resize para las tres arquitecturas
        A.Resize(384, 384),

        # 1) flips (p=0.5 cada uno para ≈50 % prob.)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # 2) rotaciones 0/90/180/270
        A.RandomRotate90(p=0.5),

        # 3) transpose (reflejo diagonal)
        A.Transpose(p=0.5),

        # 4) distortion suave
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    ])

def get_validation_augmentation():
    """Resize sólo a 384×384 sin más augmentación."""
    return A.Compose([
        A.Resize(384, 384)
    ])


# --- Funciones auxiliares ---
def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def denormalize(x):
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    return x.clip(0, 1)


# Normalización oficial de EfficientNet-B3: de [0,255] → [-1,1]
preprocess_input = sm.get_preprocessing('efficientnetb3')

class Dataset:
    CLASSES = ['corrosion']

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, i) for i in self.ids]
        self.masks_fps = [os.path.join(masks_dir, i) for i in self.ids]
        self.class_values = [self.CLASSES.index(c.lower()) for c in classes]
        self.augmentation = augmentation

    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(self.images_fps[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float32')
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        image = image.astype('float32')
        mask = mask.astype('float32')
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image  = preprocess_input(image .astype('float32'))       ##normalization
        
        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, idx):
        batch_ids = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = [self.dataset[i] for i in batch_ids]
        return tuple(np.stack(samples, axis=0) for samples in zip(*batch))

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.seed(42)
            self.indexes = np.random.permutation(self.indexes)


if __name__ == '__main__':
    ###########
    # dry-run: entrena solo 1 batch y sale (para aislar inicialización)
    dry_run = True

    # crea tu modelo, loaders, callbacks, etc…
    monitor_cb = ProgressMonitor(batch_print_freq=1)

    if dry_run:
        print("=== DRY RUN: 1 batch ===")
        model.fit(train_loader, steps_per_epoch=1, epochs=1,
                  callbacks=[monitor_cb], validation_data=valid_loader,
                  validation_steps=1)
        exit(0)

    # si dry_run=False, corre el entrenamiento completo:
    history = model.fit(
        train_loader,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        callbacks=[cp, reduce_lr_cb, monitor_cb],
        validation_data=valid_loader,
        validation_steps=len(valid_loader),
    )
###########

    # Callbacks comunes
    MODEL_DIR = os.path.join(os.getcwd(), 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, MODEL_NAME + '_{arch}.h5'),
        save_weights_only=True,
        save_best_only=True,
        mode='min'
    )
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau()
    monitor_cb = ProgressMonitor(batch_print_freq=25)

    # Datasets y dataloaders
    train_ds = Dataset(x_train_dir, y_train_dir, classes=CLASSES, augmentation=get_training_augmentation())

    valid_ds = Dataset(x_valid_dir, y_valid_dir, classes=CLASSES, augmentation=get_validation_augmentation())
    test_ds  = Dataset(x_test_dir,  y_test_dir,  classes=CLASSES, augmentation=get_validation_augmentation())

    train_loader = Dataloder(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = Dataloder(valid_ds, batch_size=4, shuffle=False)
    test_loader  = Dataloder(test_ds,  batch_size=4, shuffle=False)

    ARCHITECTURES = ['baseline', 'pspnet', 'fpn']
    results = {}

    for arch in ARCHITECTURES:
        logging.info(f"Entrenando arquitectura: {arch}")
        model = build_model(arch, BACKBONE, n_classes, activation, LR, input_shape=INPUT_SHAPE)

        # Ajuste específico del checkpoint path
        cp = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, MODEL_NAME + f'_{arch}.h5'),
            save_weights_only=True,
            save_best_only=True,
            mode='min'
        )
        logging.info(f"Empezando entrenamiento con EPOCHS = {EPOCHS}")

        # --- Diagnóstico rápido ---
        # ① Comprobar shapes de un batch de validación
        x_val, y_val = next(iter(valid_loader))
        print("SHAPES  validación:", x_val.shape, y_val.shape)

        # ② Contar cuántas máscaras tienen corrosión (>0)
        non_empty = sum(y_val.sum(axis=(1,2,3)) > 0)
        print(f"Máscaras con corrosión en el batch: {non_empty}/{len(y_val)}")





        history = model.fit(
            train_loader,
            steps_per_epoch=len(train_loader),
            epochs=EPOCHS,
            callbacks=[cp, reduce_lr_cb, monitor_cb],
            validation_data=valid_loader,
            validation_steps=len(valid_loader)
        )

        # Guardar métricas y pesos
        results[arch] = {
            'history': history.history,
            'eval_val': model.evaluate(valid_loader, verbose=0),
            'eval_test': None
        }

        # Gráficos de entrenamiento
       # fig, axes = plt.subplots(1, 2, figsize=(30, 5))
       # axes[0].plot(history.history['iou_score'], label='Train IoU')
       # axes[0].plot(history.history['val_iou_score'], label='Val IoU')
       # axes[0].set_title(f'IoU - {arch}')
       # axes[0].legend()

       # axes[1].plot(history.history['loss'], label='Train Loss')
       # axes[1].plot(history.history['val_loss'], label='Val Loss')
       # axes[1].set_title(f'Loss - {arch}')
       # axes[1].legend()

       # out_dir = os.path.join(os.getcwd(), 'graficos')
       # os.makedirs(out_dir, exist_ok=True)
       # fig.savefig(os.path.join(out_dir, f"{MODEL_NAME}_{arch}.png"))
       # plt.close(fig)

        # Evaluación en test
        model.load_weights(os.path.join(MODEL_DIR, f"{MODEL_NAME}_{arch}.h5"))
        eval_test = model.evaluate(test_loader, verbose=0)
        results[arch]['eval_test'] = eval_test

        # Guardar DataFrame de IoU
        df = pd.DataFrame({
            'Train_IoU': history.history['iou_score'],
            'Val_IoU':   history.history['val_iou_score']
        })
        excel_dir = os.path.join(os.getcwd(), 'callbacks')
        os.makedirs(excel_dir, exist_ok=True)
        df.to_excel(os.path.join(excel_dir, f"{MODEL_NAME}_{arch}.xlsx"), index=False)

        # Visualización de ejemplos de test
       # for i in 1: #range(len(test_ds))
       #     img, gt = test_ds[i]
       #     pr = model.predict(np.expand_dims(img, 0)).round()[0]
       #     visualize(a
       #         image=denormalize(img),a
       #         gt_mask=gt[..., 0],
       #         pr_mask=pr[..., 0]
       #     )

    # Imprimir resumen
    for arch, res in results.items():
        logging.info(f"{arch}: Val Loss={res['eval_val'][0]:.4f}, Val IoU={res['eval_val'][1]:.4f}, Test IoU={res['eval_test'][1]:.4f}")
