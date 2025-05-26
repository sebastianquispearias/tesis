import os

os.environ["SM_FRAMEWORK"] = "tf.keras"


import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models as sm
import cv2
from tensorflow import keras
import tensorflow as tf
from callbacks_monitor import ProgressMonitor
from models import build_model
import albumentations as A


# 0) Opciones de TF (si quieres quitar logs de XLA, descomenta)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_XLA_FLAGS'] = '--xla_hlo_profile'

tf.config.run_functions_eagerly(True)
# Permitir que TensorFlow vaya pidiendo memoria GPU según la necesite, en lugar de reservarla toda.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# ────────────────────────────────────────────────────────────────
#  ) Funciones auxiliares (incluye visualize y denormalize)
# ────────────────────────────────────────────────────────────────
def visualize(**images):
    """
    Llama a plt.imshow para cada imagen con su título.
    Puedes usarla al final para inspeccionar resultados.
    """
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([]); plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def denormalize(x):
    """De [0,255]→[-1,1] a [0,1] para visualizar imágenes."""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    return x.clip(0, 1)

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
#C:\Users\User\Desktop\tesis\data\processed_binary_384
#C:\Users\User\Desktop\tesis\data\320x320
# ──────────────────────────────────────────────────────────────────────────────
# B) PARÁMETROS GLOBALES
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR    = r'C:\Users\User\Desktop\tesis\data\320x320'
import os
print("DATA_DIR =", DATA_DIR)
print("Train images:", len(os.listdir(os.path.join(DATA_DIR, "Train", "images"))))
print("Train masks: ", len(os.listdir(os.path.join(DATA_DIR, "Train", "masks"))))

MODEL_NAME  = 'Unet_EfficientnetB3_final'
BACKBONE    = 'efficientnetb3'
BATCH_SIZE  = 1
CLASSES     = ['corrosion']
LR          = 1e-4
EPOCHS      = 1
INPUT_SHAPE = (384, 384, 3)

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)
activation = 'sigmoid' if n_classes == 1 else 'softmax'

# Rutas de carpetas
x_train_dir = os.path.join(DATA_DIR, 'Train', 'images')
y_train_dir = os.path.join(DATA_DIR, 'Train', 'masks')
x_valid_dir = os.path.join(DATA_DIR, 'Val',   'images')
y_valid_dir = os.path.join(DATA_DIR, 'Val',   'masks')
x_test_dir  = os.path.join(DATA_DIR, 'Test',  'images')
y_test_dir  = os.path.join(DATA_DIR, 'Test',  'masks')


# ──────────────────────────────────────────────────────────────────────────────
# C) AUGMENTATIONS Y PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# C) AUGMENTATIONS Y PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
import albumentations as A
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
    return A.Compose([ A.Resize(*INPUT_SHAPE[:2]) ])

def get_preprocessing(preprocessing_fn):
    """
    Envuelve preprocess_input en un Compose de Albumentations
    para aplicarlo tras la augmentación.
    """
    return A.Compose([ A.Lambda(image=preprocessing_fn, mask=None) ])


# ──────────────────────────────────────────────────────────────────────────────
# D) DATASET y DATALOADER
# ──────────────────────────────────────────────────────────────────────────────
class Dataset:
    CLASSES = CLASSES
    def __init__(self, images_dir, masks_dir, classes, augmentation=None,preprocessing=None):
        self.ids        = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, i) for i in self.ids]
        self.masks_fps  = [os.path.join(masks_dir,  i) for i in self.ids]
        self.class_values = [self.CLASSES.index(c.lower()) for c in classes]
        self.augmentation = augmentation
        self.preprocessing  = preprocessing  # <-- nuevo


    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(self.images_fps[i]), cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(self.masks_fps[i], 0)
        masks = [(mask == v) for v in self.class_values]
        mask  = np.stack(masks, axis=-1).astype('float32')
        #if self.augmentation:
        #    sample = self.augmentation(image=image, mask=mask)
        #    image, mask = sample['image'], sample['mask']
        #image = preprocess_input(image.astype('float32'))
        #return image, mask
        # 2) augmentación (solo train)
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # 3) **pre-procesamiento** igual al notebook original
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask
    
    def __len__(self):
        return len(self.ids)

class Dataloder(keras.utils.Sequence):
    def __init__(self, dataset, batch_size=1, shuffle=False):
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
            np.random.seed(42)
            np.random.shuffle(self.indexes)


# ──────────────────────────────────────────────────────────────────────────────
# E) SCRIPT PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    
    
    # 1) dry-run? (solo 1 batch para probar inicialización)
    dry_run = False

    
    # 2) crea loaders ANTES del dry-run
    train_ds = Dataset(x_train_dir, y_train_dir, classes=CLASSES, augmentation=get_training_augmentation(),preprocessing  = get_preprocessing(preprocess_input)  )
    valid_ds = Dataset(x_valid_dir, y_valid_dir, classes=CLASSES, augmentation=get_validation_augmentation(),preprocessing  = get_preprocessing(preprocess_input)  )
    train_loader = Dataloder(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = Dataloder(valid_ds, batch_size=1, shuffle=False)


    # ——— Prueba de shapes ———
    x_batch, y_batch = train_loader[0]
    print("x_batch.shape =", x_batch.shape)   # (1, 384, 384, 3)
    print("y_batch.shape =", y_batch.shape)   # (1, 384, 384, 1)
    x_val, y_val     = valid_loader[0]
    print("x_val.shape   =", x_val.shape)

    # 3) crea modelo y callback de monitor ANTES
    model = build_model('baseline', BACKBONE, n_classes, activation, LR, input_shape=INPUT_SHAPE)
    monitor_cb = ProgressMonitor(batch_print_freq=1)

    if dry_run:
        print("=== DRY RUN: 1 batch ===")
        model.fit(
            train_loader, steps_per_epoch=1, epochs=1,
            callbacks=[monitor_cb],
            validation_data=valid_loader, validation_steps=1
        )
        sys.exit(0)

    # 4) define carpeta MODELOS y callbacks **fuera** del bucle
    MODEL_DIR = os.path.join(os.getcwd(), 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)


   # cp = keras.callbacks.ModelCheckpoint(
   #     filepath=os.path.join(MODEL_DIR, MODEL_NAME + '_{arch}.h5'),
   #     save_weights_only=True,
   #     save_best_only=True,
   #     mode='min'
   # )
    reduce_lr = keras.callbacks.ReduceLROnPlateau()
    monitor_cb    = ProgressMonitor(batch_print_freq=25)

    # 5) crea loader de test
    test_ds     = Dataset(x_test_dir, y_test_dir, classes=CLASSES, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocess_input))
    test_loader = Dataloder(test_ds, batch_size=1, shuffle=False)

    # 6) bucle de arquitecturas
    ARCHITECTURES = ['baseline', 'pspnet', 'fpn']
    results = {}   # ← aquí guardamos history + evaluaciones

    for arch in ARCHITECTURES:
        logging.info(f"Entrenando arquitectura: {arch}")
        tf.keras.backend.clear_session()
        model = build_model(arch, BACKBONE, n_classes, activation, LR, input_shape=INPUT_SHAPE)
        model.summary()     # imprime la arquitectura y comprueba que no falle aquí

        # 6.1) nuevo modelo por arquitectura
        model = build_model(arch, BACKBONE, n_classes, activation, LR, input_shape=INPUT_SHAPE)
        cp = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"{MODEL_NAME}_{arch}.weights.h5"),
            save_weights_only=True,
            save_best_only=False,
            mode='min'
        )
        # 6.2) entrena
        history = model.fit(
            train_loader,
            steps_per_epoch=len(train_loader),
            epochs=EPOCHS,
            callbacks=[cp, reduce_lr , monitor_cb],  # ← usa los callbacks predefinidos
            validation_data=valid_loader,
            validation_steps=len(valid_loader)#1##########################################################################len(valid_loader)
        )
        # ¡MUY IMPORTANTE! recarga pesos antes de evaluar test:
        model.load_weights(os.path.join(MODEL_DIR, f"{MODEL_NAME}_{arch}.weights.h5"))
        test_metrics = model.evaluate(test_loader, verbose=0)

        # 6.3) guarda en results
        results[arch] = {
            'history':  history.history,
            'eval_val': model.evaluate(valid_loader, verbose=0),
            'eval_test': test_metrics
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

        # 6.4) evalúa test y guarda
        model.load_weights(os.path.join(MODEL_DIR, f"{MODEL_NAME}_{arch}.weights.h5"))
        results[arch]['eval_test'] = model.evaluate(test_loader, verbose=0)

        # 6.5) exporta métricas a Excel
        df = pd.DataFrame({
            'Train_IoU': history.history.get('iou_score', []),
            'Val_IoU':   history.history.get('val_iou_score', []),
            'Train_F1':  history.history.get('f1-score',       []),
            'Val_F1':    history.history.get('val_f1-score',   []),            
        })
        excel_dir = os.path.join(os.getcwd(), 'callbacks')
        os.makedirs(excel_dir, exist_ok=True)
        df.to_excel(os.path.join(excel_dir, f"{MODEL_NAME}_{arch}.xlsx"), index=False)


        # 6.5) **Usa visualize** para ver un ejemplo del último batch de validación
        #       Descomenta estas líneas si quieres mostrar:
        # x_val, y_val = next(iter(valid_loader))
        # preds = model.predict(x_val).round()
        # visualize(
        #     image=denormalize(x_val[0]),
        #     gt_mask=y_val[0,...,0],
        #     pr_mask=preds[0,...,0]
        # )

        
        # Visualización de ejemplos de test
       # for i in 1: #range(len(test_ds))
       #     img, gt = test_ds[i]
       #     pr = model.predict(np.expand_dims(img, 0)).round()[0]
       #     visualize(a
       #         image=denormalize(img),a
       #         gt_mask=gt[..., 0],
       #         pr_mask=pr[..., 0]
       #     )

    # 7) resumen final
    for arch, res in results.items():
        val_loss, val_iou, val_f1 = res['eval_val']
        test_loss, test_iou, test_f1 = res['eval_test']

        logging.info(
            f"{arch}: "
            f"Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}, Val F1={val_f1:.4f} | "
            f"Test Loss={test_loss:.4f}, Test IoU={test_iou:.4f}, Test F1={test_f1:.4f}"
        )


