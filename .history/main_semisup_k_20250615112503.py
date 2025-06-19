#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_final_ultimoPCnuevosemisuperv.py – Entrenamiento supervisado (A-Full, Bxx)
y semi-supervisado (Cxx, Mean Teacher)

Uso:
  python main_final_ultimoPCnuevosemisuperv.py --mode supervised --regime B25
  python main_final_ultimoPCnuevosemisuperv.py --mode semi       --regime C25
  python main_semisup_gtestaug.py --mode semi --regime C75

"""

import os
import sys
os.environ["SM_FRAMEWORK"] = "tf.keras"

import argparse
import logging
import random
import numpy as np
import tensorflow as tf
# Evita que TF reserve toda la memoria GPU de golpe
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
import cv2
import csv
from metrics_utils import evaluate_and_log

from pathlib import Path
from tensorflow import keras
from tensorflow.keras import backend as K
import segmentation_models as sm    
from sklearn.model_selection import train_test_split

from callbacks_monitor import ProgressMonitor
from models import build_model

import albumentations as A

def get_strong_augmentation(input_shape):
    """Transformaciones fuertes para el ESTUDIANTE."""
    return A.Compose([
        # 1) Redimensionar
        A.Resize(*input_shape[:2]),
        # 2) Geométricas
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        # 3) Fotométricas
        A.RandomBrightnessContrast(p=0.5),
        # 4) Enmascarado
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    ])

def get_weak_augmentation(input_shape):
    """Transformaciones débiles para el PROFESOR."""
    return A.Compose([
        A.Resize(*input_shape[:2]),
        A.HorizontalFlip(p=0.2),            # flip suave :contentReference[oaicite:0]{index=0}
        A.RandomBrightnessContrast(p=0.1),
    ])



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

ARCHITECTURES = [ 'fpn']  # usado en modo supervisado

# Parámetros Mean Teacher
EMA_ALPHA        = 0.99
CONS_MAX         = 1.0
CONS_RAMPUP      = 30
UNLABELED_WEIGHT = 1.0

# ──────────────────────────────────────────────────────────────────────────────
# C) AUGMENTATIONS Y PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
preprocess_input = sm.get_preprocessing(BACKBONE)

def get_training_augmentation():
    return A.Compose([
    # 1) Redimensionar
        A.Resize(*input_shape[:2]),
        # 2) Geométricas
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        # 3) Fotométricas
        A.RandomBrightnessContrast(p=0.5),
        # 4) Enmascarado
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
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
                 batch_size_lab=BATCH_SIZE, batch_size_unlab=BATCH_SIZE,
                 augment_lab=None, augment_unlab_student=None, augment_unlab_teacher=None, preprocess=None):

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
        self.augment_unlab_student = augment_unlab_student
        self.augment_unlab_teacher = augment_unlab_teacher
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

        x_unl_student, x_unl_teacher = [], []
        
        for fname in unl_batch_files:
            img = cv2.cvtColor(
                cv2.imread(os.path.join(self.x_unlab_dir, fname)),
                cv2.COLOR_BGR2RGB
            )

            # 1) Debes pasar la imagen RAW (variable `img`), no `img_unl` ni `img_unlab`
            if self.augment_unlab_student:
                aug_s = self.augment_unlab_student(image=img)
                img_unl_student = aug_s['image']
            if self.augment_unlab_teacher:
                aug_t = self.augment_unlab_teacher(image=img)
                img_unl_teacher = aug_t['image']
        
            # 2) Hay que PREPROCESAR ambas versiones (student y teacher), no solo una:
            if self.preprocess:
                img_unl_student = self.preprocess(image=img_unl_student)['image']
                img_unl_teacher = self.preprocess(image=img_unl_teacher)['image']
        
            x_unl_student.append(img_unl_student)
            x_unl_teacher.append(img_unl_teacher)

        x_unl_student = np.stack(x_unl_student, axis=0)
        x_unl_teacher = np.stack(x_unl_teacher, axis=0)
        return x_lab, y_lab, x_unl_student, x_unl_teacher

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

    best_iou   = 0.0
    best_epoch = 0

    # ───── LÓGICA “SUPERVISED” (A-Full y Bxx) ─────
    if mode == "supervised":
        best_path = os.path.join(MODEL_DIR, f"{arch}_best_supervised.weights.h5")

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

            # Construye el modelo, congelando encoder para B-regimes
            freeze_enc = regime.startswith("BB")   # True si es B10/B25/B50/B75
            model = build_model(
                arch, BACKBONE, n_classes, activation, LR,
                input_shape=INPUT_SHAPE,
                freeze_encoder=freeze_enc
            )
            model.summary()
            # ——— LOGGING EXPLÍCITO DE FREEZE ENCODER ———
            # Imprime el flag
            logging.info(f"freeze_encoder={freeze_enc}")
            # Cuenta parámetros
            trainable_count     = int(np.sum([K.count_params(w) for w in model.trainable_weights]))
            non_trainable_count = int(np.sum([K.count_params(w) for w in model.non_trainable_weights]))
            logging.info(f"Trainable params: {trainable_count:,} | Non-trainable params: {non_trainable_count:,}")
            # ————————————————————————————————————————

            # Callbacks: checkpoint y reduce_lr + progress monitor
            MODEL_DIR = os.path.join(os.getcwd(), f"models_supervised_{arch}")
            os.makedirs(MODEL_DIR, exist_ok=True)
            history_path = os.path.join(MODEL_DIR, f"{arch}_MT_history_{regime}.csv")
            best_path = os.path.join(MODEL_DIR, f"{arch}_best_supervised.weights.h5")

            cp = keras.callbacks.ModelCheckpoint(
                filepath=best_path,
                save_weights_only=True,
                save_best_only=True,
                save_freq='epoch',
                verbose=1,
                monitor='val_iou',
                mode='max'
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
            model.load_weights(best_path)


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
        # Definir las augmentaciones aquí (antes de usar el DataLoader)
        strong_aug = get_strong_augmentation(INPUT_SHAPE)
        weak_aug   = get_weak_augmentation(INPUT_SHAPE)

        # Crear el DataLoader con la augmentación diferente para el estudiante y el profesor
        train_mixed_loader = MixedDataLoader(
            x_lab_dir=x_train_lab_dir,
            y_lab_dir=y_train_lab_dir,
            x_unlab_dir=x_train_unlab_dir,
            batch_size_lab=BATCH_SIZE,
            batch_size_unlab=BATCH_SIZE,
            augment_lab=strong_aug,  
            augment_unlab_student=strong_aug,
            augment_unlab_teacher=weak_aug, 
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
            history_path = os.path.join(MODEL_DIR, f"{arch}_MT_history_{regime}.csv")
            with open(history_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Training loop manual
            best_iou   = 0.0
            best_epoch = 0
            best_path  = os.path.join(MODEL_DIR, f"{arch}_best_student.weights.h5")

            for epoch in range(EPOCHS):
                logging.info(f"--- Época {epoch+1}/{EPOCHS} (Mean Teacher) ---")
                cons_w = get_consistency_weight(epoch)

                for x_lab, y_lab, x_unl_s, x_unl_t in train_mixed_loader:
                    # → Teacher sobre versión débil
                    t_preds = teacher(x_unl_t, training=False)
                    t_probs = tf.sigmoid(t_preds)

                    with tf.GradientTape() as tape:
                        # Student en labeled
                        s_lab_preds = student(x_lab, training=True)
                        s_lab_probs = tf.sigmoid(s_lab_preds)
                        
                        if s_lab_probs.dtype == tf.float16:
                            s_lab_probs = tf.cast(s_lab_probs, tf.float32)
                        
                        # Student en unlabeled
                        s_unl_preds = student(x_unl_s, training=True)
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

                # ────────────────────────────────────────────────────────────
                # ▼ EN LUGAR de recorrer val_loader manualmente, llamamos a evaluate_and_log
                try:
                    val_loss, val_iou, val_prec, val_rec, val_f1 = evaluate_and_log(
                        student,      # tu modelo student
                        val_loader,   # tu DataLoader de validación
                        writer,       # CSV writer
                        epoch,        # número de época
                        loss_sup,     # pérdida supervisada calculada en el batch
                        loss_cons,    # pérdida de consistencia
                        loss_total    # suma de las dos anteriores
                    )                        
                    logging.info(
                        f"Época {epoch}: sup={loss_sup:.4f}, cons={loss_cons:.4f}, total={loss_total:.4f} | "
                        f"val_loss={val_loss:.4f}, IoU={val_iou:.4f}, P={val_prec:.4f}, "
                        f"R={val_rec:.4f}, F1={val_f1:.4f}"
                    )

                except Exception as e:
                    # Alerta si evaluate_and_log falla por algún motivo
                    print(f"[main_semisup_g] ERROR en evaluate_and_log en época {epoch+1}: {e}")
                    val_loss, val_iou = float('nan'), float('nan')

                logging.info(
                    f"{arch} (MT) Época {epoch+1}: Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}"
                )
                # ────
                # ─────────────────────────────────────────────
                # Guardar sólo si esta época supera el mejor IoU_val
                if val_iou > best_iou:
                    best_iou   = val_iou
                    best_epoch = epoch + 1
                    student.save_weights(best_path)
                    logging.info(f"[semi] ▶ Nuevo best IoU_val={best_iou:.4f} (época {best_epoch})")

                # Evaluación final en test (student)
# ─────────────────────────────────────────────
                # Carga el mejor modelo según validación
                student.load_weights(best_path)
                logging.info(f"[semi] ✔ Cargado best checkpoint época {best_epoch} con IoU_val={best_iou:.4f}")

                test_metrics = student.evaluate(test_loader, verbose=0)
                test_loss, test_iou = test_metrics[0], test_metrics[1]
                logging.info(
                    f"{arch} (MT) Test final: Loss={test_metrics[0]:.4f}, IoU={test_metrics[1]:.4f}"
                )
                       

    else:
        raise ValueError("Modo no reconocido: use --mode supervised o --mode semi")

    print("¡Entrenamiento completado!")
