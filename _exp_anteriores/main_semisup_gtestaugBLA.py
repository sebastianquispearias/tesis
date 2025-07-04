#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_final_ultimoPCnuevosemisuperv.py – Entrenamiento supervisado (A-Full, Bxx)
y semi-supervisado (Cxx, Mean Teacher)

Uso:
  python main_final_ultimoPCnuevosemisuperv.py --mode supervised --regime B25
  python main_final_ultimoPCnuevosemisuperv.py --mode semi       --regime C25
"""

import os
import sys
os.environ["SM_FRAMEWORK"] = "tf.keras"

import argparse
import logging
import random
import numpy as np
import tensorflow as tf
import cv2

from pathlib import Path
from tensorflow import keras
import segmentation_models as sm    
from sklearn.model_selection import train_test_split

from callbacks_monitor import ProgressMonitor
from models import build_model

import albumentations as A

# ──────────────────────────────────────────────────────────────────────────────
# 0) OPCIONES DE TF y SEEDING
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.run_functions_eagerly(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ──────────────────────────────────────────────────────────────────────────────
# A) CONFIGURACIÓN DE LOGGING
# ──────────────────────────────────────────────────────────────────────────────
log = open('entrenamiento_log.txt', 'w', encoding='utf-8')
sys.stdout = log
sys.stderr = log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(log)]
)

# ──────────────────────────────────────────────────────────────────────────────
# B) PARÁMETROS GLOBALES
# ──────────────────────────────────────────────────────────────────────────────
BACKBONE    = 'efficientnetb3'
BATCH_SIZE  = 4
CLASSES     = ['corrosion']
LR          = 1e-4
EPOCHS      = 80
INPUT_SHAPE = (384, 384, 3)
n_classes   = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)
activation  = 'sigmoid' if n_classes == 1 else 'softmax'

ARCHITECTURES = ['baseline', 'pspnet', 'fpn']  # usado en modo supervisado

# Parámetros Mean Teacher
EMA_ALPHA        = 0.99
CONS_MAX         = 1.0
CONS_RAMPUP      = 20
UNLABELED_WEIGHT = 1.0

# ──────────────────────────────────────────────────────────────────────────────
# C) AUGMENTATIONS Y PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
preprocess_input = sm.get_preprocessing(BACKBONE)

def get_training_augmentation():
    return A.Compose([
        A.Resize(*INPUT_SHAPE[:2]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    ])

def get_validation_augmentation():
    return A.Compose([
        A.Resize(*INPUT_SHAPE[:2]),
    ])

def get_preprocessing(preprocessing_fn):
    return A.Compose([
        A.Lambda(image=preprocessing_fn)
    ])

# ──────────────────────────────────────────────────────────────────────────────
# D) DATASET y DATALOADER para imágenes etiquetadas
# ──────────────────────────────────────────────────────────────────────────────
class Dataset:
    CLASSES = CLASSES
    def __init__(self, images_dir, masks_dir, classes,
                 augmentation=None, preprocessing=None):
        self.ids        = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, i) for i in self.ids]
        self.masks_fps  = [os.path.join(masks_dir,  i) for i in self.ids]
        self.class_values = [self.CLASSES.index(c.lower()) for c in classes]
        self.augmentation  = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        img_path = self.images_fps[i]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        mask_path = self.masks_fps[i]
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (raw_mask > 0).astype('float32')[..., None]

        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        if self.preprocessing:
            processed = self.preprocessing(image=image, mask=mask)
            image, mask = processed['image'], processed['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

class Dataloder(keras.utils.Sequence):
    def __init__(self, dataset, batch_size=BATCH_SIZE, shuffle=False):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.indexes    = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, idx):
        batch_ids = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch = [self.dataset[i] for i in batch_ids]
        return tuple(np.stack(s, axis=0) for s in zip(*batch))

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.seed(SEED)
            np.random.shuffle(self.indexes)

# ──────────────────────────────────────────────────────────────────────────────
# E) MIXED DATALOADER para Mean Teacher (etiquetados + no-etiquetados)
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# E) MIXED DATALOADER para Mean Teacher (etiquetados + no-etiquetados)
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# E) MIXED DATALOADER para Mean Teacher (etiquetados + no-etiquetados)
# ──────────────────────────────────────────────────────────────────────────────
class MixedDataLoader(keras.utils.Sequence):
    """
    Retorna en cada paso: x_lab_batch, y_lab_batch, x_unl_batch.
    Ahora __len__ = número de lotes basados en imágenes etiquetadas,
    y comprueba que las carpetas no estén vacías al inicio.
    """
    def __init__(self, x_lab_dir, y_lab_dir, x_unlab_dir,
                 batch_size_lab=4, batch_size_unlab=4,
                 augment_lab=None, augment_unlab=None, preprocess=None):

        self.lab_images = sorted(os.listdir(x_lab_dir))
        self.unl_images = sorted(os.listdir(x_unlab_dir))

        # ── Comprobaciones tempranas ──
        if len(self.lab_images) == 0:
            raise RuntimeError(f"ERROR: no se encontraron imágenes etiquetadas en:\n  {x_lab_dir}")
        if len(self.unl_images) == 0:
            raise RuntimeError(f"ERROR: no se encontraron imágenes sin etiqueta en:\n  {x_unlab_dir}")

        self.x_lab_dir   = x_lab_dir
        self.y_lab_dir   = y_lab_dir
        self.x_unlab_dir = x_unlab_dir

        self.batch_size_lab   = batch_size_lab
        self.batch_size_unlab = batch_size_unlab
        self.augment_lab      = augment_lab
        self.augment_unlab    = augment_unlab
        self.preprocess       = preprocess

        self.on_epoch_end()

    def __len__(self):
        # Número de lotes basados únicamente en la parte etiquetada
        n_lab = len(self.lab_images) // self.batch_size_lab
        return max(1, n_lab)

    def on_epoch_end(self):
        np.random.shuffle(self.lab_images)
        np.random.shuffle(self.unl_images)

    def __getitem__(self, idx):
        # ◀── Si idx ya no está en [0, len(self)-1], cortamos la iteración
        if idx >= self.__len__():
            raise IndexError

        # —— Lote etiquetado —— 
        start_lab = idx * self.batch_size_lab
        end_lab   = start_lab + self.batch_size_lab
        lab_batch_files = self.lab_images[start_lab:end_lab]

        x_lab = []
        y_lab = []
        for fname in lab_batch_files:
            img = cv2.cvtColor(
                cv2.imread(os.path.join(self.x_lab_dir, fname)),
                cv2.COLOR_BGR2RGB
            )
            mask = cv2.imread(
                os.path.join(self.y_lab_dir, fname),
                cv2.IMREAD_GRAYSCALE
            )
            mask = (mask > 0).astype('float32')[..., None]

            if self.augment_lab:
                aug = self.augment_lab(image=img, mask=mask)
                img, mask = aug['image'], aug['mask']
            if self.preprocess:
                pr = self.preprocess(image=img, mask=mask)
                img, mask = pr['image'], pr['mask']

            x_lab.append(img)
            y_lab.append(mask)

        x_lab = np.stack(x_lab, axis=0)
        y_lab = np.stack(y_lab, axis=0)

        # —— Lote sin-etiquetar —— 
        start_unl = idx * self.batch_size_unlab
        end_unl   = start_unl + self.batch_size_unlab
        unl_batch_files = self.unl_images[start_unl:end_unl]

        x_unl = []
        for fname in unl_batch_files:
            img = cv2.cvtColor(
                cv2.imread(os.path.join(self.x_unlab_dir, fname)),
                cv2.COLOR_BGR2RGB
            )
            if self.augment_unlab:
                aug = self.augment_unlab(image=img)
                img = aug['image']
            if self.preprocess:
                pr = self.preprocess(image=img, mask=None)
                img = pr['image']
            x_unl.append(img)

        x_unl = np.stack(x_unl, axis=0)
        return x_lab, y_lab, x_unl


# ──────────────────────────────────────────────────────────────────────────────
# F) PARSEO DE ARGUMENTOS
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["supervised", "semi"],
        required=True,
        help=(
            "supervised → entrenar con model.fit (A-Full y Bxx);\n"
            "semi       → entrenar con Mean Teacher (Cxx)."
        )
    )
    p.add_argument(
        "--regime",
        choices=["A-Full", "B10", "B25", "B50", "B75",
                 "C10", "C25", "C50", "C75"],
        required=True,
        help="Régimen de datos (carpeta dentro de data/all_results/regimes)."
    )
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# G) FUNCIÓN PARA RAMP-UP DE CONSISTENCY WEIGHT
# ──────────────────────────────────────────────────────────────────────────────
def get_consistency_weight(epoch):
    if epoch >= CONS_RAMPUP:
        return CONS_MAX
    phase = 1.0 - epoch / CONS_RAMPUP
    return CONS_MAX * np.exp(-5 * phase * phase)

# ──────────────────────────────────────────────────────────────────────────────
# H) MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    args   = parse_args()
    mode   = args.mode      # “supervised” o “semi”
    regime = args.regime    # p.ej. “B25” o “C25”

    # Definir DATA_DIR según régimen
    DATA_ROOT = r"C:\Users\User\Desktop\tesis\data\all_results\regimes"
    DATA_DIR  = os.path.join(DATA_ROOT, regime)

    # Rutas de entrenamiento, validación y test
    x_train_lab_dir   = os.path.join(DATA_DIR, 'train', 'images_labeled')
    y_train_lab_dir   = os.path.join(DATA_DIR, 'train', 'masks_labeled')
    x_train_unlab_dir = os.path.join(DATA_DIR, 'train', 'images_unlabeled')
    x_val_dir         = os.path.join(DATA_DIR, 'val',   'images')
    y_val_dir         = os.path.join(DATA_DIR, 'val',   'masks')
    x_test_dir        = os.path.join(DATA_DIR, 'test',  'images')
    y_test_dir        = os.path.join(DATA_DIR, 'test',  'masks')

    # ───── LÓGICA “SUPERVISED” (A-Full y Bxx) ─────
    if mode == "supervised":
        # Crear datasets y loaders para labeled-only
        train_ds = Dataset(
            x_train_lab_dir, y_train_lab_dir, classes=CLASSES,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocess_input)
        )
        valid_ds = Dataset(
            x_val_dir, y_val_dir, classes=CLASSES,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocess_input)
        )
        train_loader = Dataloder(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = Dataloder(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Crear loader de test
        test_ds    = Dataset(
            x_test_dir, y_test_dir, classes=CLASSES,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocess_input)
        )
        test_loader = Dataloder(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Bucle sobre arquitecturas
        for arch in ARCHITECTURES:
            logging.info(f"Entrenando arquitectura (supervised): {arch}")
            tf.keras.backend.clear_session()

            model = build_model(arch, BACKBONE, n_classes, activation, LR,
                                input_shape=INPUT_SHAPE)
            model.summary()

            # Callbacks: checkpoint y reduce_lr + progress monitor
            MODEL_DIR = os.path.join(os.getcwd(), f"models_supervised_{arch}")
            os.makedirs(MODEL_DIR, exist_ok=True)

            cp = keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(MODEL_DIR, f"{arch}.weights.h5"),
                save_weights_only=True,
                save_best_only=False,
                save_freq='epoch',
                verbose=1,
                monitor='val_loss',
                mode='min'
            )
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, verbose=1
            )
            monitor_cb = ProgressMonitor(batch_print_freq=25)

            # Entrenar
            model.fit(
                train_loader,
                steps_per_epoch=len(train_loader),
                epochs=EPOCHS,
                callbacks=[cp, reduce_lr, monitor_cb],
                validation_data=valid_loader,
                validation_steps=len(valid_loader)
            )

            # Cargar pesos antes de evaluar test
            weight_path = os.path.join(MODEL_DIR, f"{arch}.weights.h5")
            if os.path.exists(weight_path):
                model.load_weights(weight_path)
            else:
                raise FileNotFoundError(f"Pesos no encontrados: {weight_path}")

            # Evaluar
            val_metrics  = model.evaluate(valid_loader, verbose=0)
            test_metrics = model.evaluate(test_loader,  verbose=0)

            logging.info(
                f"{arch} (supervised): "
                f"Val Loss={val_metrics[0]:.4f}, Val IoU={val_metrics[1]:.4f} | "
                f"Test Loss={test_metrics[0]:.4f}, Test IoU={test_metrics[1]:.4f}"
            )

    # ───── LÓGICA “SEMI” (Mean Teacher para Cxx) ─────
    elif mode == "semi":
        # MixedDataLoader para train (labeled + unlabeled)
        train_mixed_loader = MixedDataLoader(
            x_lab_dir=x_train_lab_dir,
            y_lab_dir=y_train_lab_dir,
            x_unlab_dir=x_train_unlab_dir,
            batch_size_lab=4,
            batch_size_unlab=4,
            augment_lab=get_training_augmentation(),
            augment_unlab=get_training_augmentation(),
            preprocess=get_preprocessing(preprocess_input)
        )

        # Validación y test: student usa Dataset/Dataloder normales
        val_loader = Dataloder(
            Dataset(x_val_dir, y_val_dir, classes=CLASSES,
                    augmentation=get_validation_augmentation(),
                    preprocessing=get_preprocessing(preprocess_input)),
            batch_size=BATCH_SIZE, shuffle=False
        )
        test_loader = Dataloder(
            Dataset(x_test_dir, y_test_dir, classes=CLASSES,
                    augmentation=get_validation_augmentation(),
                    preprocessing=get_preprocessing(preprocess_input)),
            batch_size=BATCH_SIZE, shuffle=False
        )

        # Entrenar para cada arquitectura
        for arch in ARCHITECTURES:
            logging.info(f"Entrenando arquitectura (Mean Teacher): {arch}")
            tf.keras.backend.clear_session()

            # Construir student y teacher
            student = build_model(arch, BACKBONE, n_classes, activation, LR,
                                  input_shape=INPUT_SHAPE)
            teacher  = build_model(arch, BACKBONE, n_classes, activation, LR,
                                   input_shape=INPUT_SHAPE)
            teacher.set_weights(student.get_weights())
            teacher.trainable = False

            optimizer = keras.optimizers.Adam(LR)
            dice_loss = sm.losses.DiceLoss()
            mse_loss  = tf.keras.losses.MeanSquaredError()

            MODEL_DIR = os.path.join(os.getcwd(), f"models_MT_{arch}")
            os.makedirs(MODEL_DIR, exist_ok=True)
            # Training loop manual
            for epoch in range(EPOCHS):
                logging.info(f"--- Época {epoch+1}/{EPOCHS} (Mean Teacher) ---")
                cons_w = get_consistency_weight(epoch)

                for x_lab, y_lab, x_unl in train_mixed_loader:
                    # Teacher predice en x_unl sin gradiente
                    t_preds = teacher(x_unl, training=False)
                    t_probs = tf.sigmoid(t_preds)

                    with tf.GradientTape() as tape:
                        # Student en labeled
                        s_lab_preds = student(x_lab, training=True)
                        s_lab_probs = tf.sigmoid(s_lab_preds)
                        # Student en unlabeled
                        s_unl_preds = student(x_unl, training=True)
                        s_unl_probs = tf.sigmoid(s_unl_preds)

                        loss_sup  = dice_loss(y_lab, s_lab_probs)
                        loss_cons = mse_loss(t_probs, s_unl_probs)
                        loss_total = loss_sup + UNLABELED_WEIGHT * cons_w * loss_cons

                    grads = tape.gradient(loss_total, student.trainable_variables)
                    optimizer.apply_gradients(zip(grads, student.trainable_variables))

                    # Actualizar teacher con EMA
                    sw = student.get_weights()
                    tw = teacher.get_weights()
                    new_tw = [
                        EMA_ALPHA * tw_i + (1.0 - EMA_ALPHA) * sw_i
                        for sw_i, tw_i in zip(sw, tw)
                    ]
                    teacher.set_weights(new_tw)

                # Evaluación en validación (solo student)
                val_metrics = student.evaluate(val_loader, verbose=0)
                val_loss, val_iou = val_metrics[0], val_metrics[1]
                logging.info(
                    f"{arch} (MT) Época {epoch+1}: Val Loss={val_metrics[0]:.4f}, "
                    f"Val IoU={val_metrics[1]:.4f}"
                )
                # Guardar checkpoint student
                student.save_weights(os.path.join(
                    MODEL_DIR, f"{arch}_student_epoch{epoch+1}.weights.h5"
                ))

            # Evaluación final en test (student)
            test_metrics = student.evaluate(test_loader, verbose=0)
            test_loss, test_iou = test_metrics[0], test_metrics[1]
            logging.info(
                f"{arch} (MT) Test final: Loss={test_metrics[0]:.4f}, IoU={test_metrics[1]:.4f}"
            )
                       

    else:
        raise ValueError("Modo no reconocido: use --mode supervised o --mode semi")

    print("¡Entrenamiento completado!")
