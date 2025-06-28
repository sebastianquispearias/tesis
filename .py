# --- Cell 0 ---
#!pip install -q tensorflow==2.4.1
#!pip install -q keras==2.4.0
import sys
from callbacks_monitor import ProgressMonitor

log = open('entrenamiento_log.txt', 'w', encoding ='utf-8')
sys.stdout = log
sys.stderr = log
# --- Cell 1 ---
import tensorflow as tf
print(tf.__version__)

# --- Cell 2 ---
#!pip install python-git
#!pip install keras-applications
#!pip install image-classifiers
#!pip install efficientnet
#!pip install segmentation-models
#!pip install git+https://github.com/qubvel/segmentation_models
#!pip install -U albumentations==1.0.2 --user
#!pip install -U git+https://github.com/albu/albumentations
#!pip install -U albumentations[imgaug]

# --- Cell 3 ---
from tensorflow.python.client import device_lib

device_lib.list_local_devices()

# --- Cell 4 ---
#from google.colab import drive
#drive.mount('/content/drive')

# --- Cell 5 ---
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
from tensorflow import keras
#from keras.utils import generic_utils
import segmentation_models as sm
#sm.set_framework('keras')
keras.backend.set_image_data_format('channels_last')


import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import random

# --- Cell 6 ---
DATA_DIR = r'C:\Users\User\Desktop\tesis\data\320x320'

MODEL_NAME='Unet_EfficientnetB3_final'
BACKBONE = 'efficientnetb3'
BATCH_SIZE = 8
CLASSES = ['corrosion']
LR = 0.0001
EPOCHS = 80

preprocess_input = sm.get_preprocessing(BACKBONE)

# --- Cell 7 ---
x_train_dir = os.path.join(DATA_DIR, 'Train', 'images')
y_train_dir = os.path.join(DATA_DIR, 'Train', 'masks')

x_valid_dir = os.path.join(DATA_DIR, 'Val', 'images')
y_valid_dir = os.path.join(DATA_DIR, 'Val', 'masks')

x_test_dir  = os.path.join(DATA_DIR, 'Test', 'images')
y_test_dir  = os.path.join(DATA_DIR, 'Test', 'masks')

# --- Cell 8 ---
# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['corrosion']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = None
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        # add background if mask is not binary
        # 1) crea la máscara directamente en float32
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float32')

        # 2) añade fondo si hiciera falta
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # 3) tras augment/preprocessing, forzamos dtype float32 en imagen y máscara:
        image = image.astype('float32')
        mask  = mask.astype('float32')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return tuple(batch)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        np.random.seed(42)
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

# --- Cell 9 ---
# Lets look at data we have
dataset = Dataset(x_train_dir, y_train_dir, classes=['corrosion'])

image, mask = dataset[5] # get some sample
visualize(
    image=image,
    corrosion_mask=mask[..., 0].squeeze(),
)


# --- Cell 10 ---
def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    preprocess_input = sm.get_preprocessing(BACKBONE)

# --- Cell 11 ---
# Lets look at augmented data we have
dataset = Dataset(x_train_dir, y_train_dir, classes=['corrosion'], augmentation=None)

image, mask = dataset[12] # get some sample
visualize(
    image=image,
    corrosion_mask=mask[..., 0].squeeze(),
)

# --- Cell 12 ---
import segmentation_models as sm

# segmentation_models could also use `tf.keras` if you do not have Keras installed
# or you could switch to other framework using `sm.set_framework('tf.keras')`

# --- Cell 13 ---
# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, encoder_weights='imagenet')
model_input = model.input
model_output = model.get_layer('final_conv').output #(any layer you want)
#add dropout
model_output = keras.layers.Dropout(0.3)(model_output)
#add activation
output = keras.layers.Activation(activation, name=activation)(model_output)
model_dp = keras.models.Model(model_input, output)
model=model_dp

# --- Cell 14 ---
# define optomizer
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
#dice_loss = sm.losses.DiceLoss()
#focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = sm.losses.DiceLoss()

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
# model.compile(optim, total_loss, metrics)

model.compile(optimizer=optim,loss=total_loss,metrics=metrics,loss_weights=None)




# --- Cell 15 ---
model.summary()

# --- Cell 16 ---

# Creamos nuestros callbacks locales
MODEL_DIR = r"C:\Users\User\Desktop\tesis\models"
os.makedirs(MODEL_DIR, exist_ok=True)

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, MODEL_NAME + '.h5'),
    save_weights_only=True,
    save_best_only=True,
    mode='min'
)
reduce_lr_cb = keras.callbacks.ReduceLROnPlateau()
monitor_cb   = ProgressMonitor(batch_print_freq=25)

callbacks_list = [
    checkpoint_cb,
    reduce_lr_cb,
    monitor_cb,
]

# Dataset for train images

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    classes=CLASSES,
    augmentation=None,
    preprocessing=None,
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    classes=CLASSES,
    augmentation=None,
    preprocessing=None,
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, 320, 320, n_classes)

# define callbacks for learning rate scheduling and best checkpoints saving
#callbacks = [
#    keras.callbacks.ModelCheckpoint('/content/drive/My Drive/Segmentation_Models_Colab/models/'+MODEL_NAME+'.weights.h5', save_weights_only=True, save_best_only=True, mode='min'),
#    keras.callbacks.ReduceLROnPlateau(),
#]

# --- Cell 17 ---
#Lets look at validation data we have
image, mask = valid_dataset[12] # get some sample
visualize(
    image=image,
    corrosion_mask=mask[..., 0].squeeze(),
)

# --- Cell 18 ---
import time
import pandas as pd
start = time.time()

# --- Cell 19 ---
# train model
history = model.fit(
    train_dataloader,
    steps_per_epoch=len(train_dataloader),
    epochs=EPOCHS,
    callbacks=callbacks_list,
    validation_data=valid_dataloader,
    validation_steps=len(valid_dataloader),
)

# --- Cell 20 ---
# Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
OUT_DIR = r"C:\Users\User\Desktop\tesis\graficos"
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUT_DIR, MODEL_NAME + '.PNG'))

plt.show()


# --- Cell 21 ---
test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    classes=CLASSES,
    augmentation=None,
    preprocessing=None,
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

# --- Cell 22 ---
#Lets look at test data we have

image, mask = test_dataset[9] # get some sample
visualize(
    image=image,
    corrosion_mask=mask[..., 0].squeeze(),
)

# --- Cell 23 ---
# load best weights
model.load_weights(os.path.join(MODEL_DIR, MODEL_NAME + '.h5'))

# --- Cell 24 ---

scores = model.evaluate_generator(test_dataloader)

print("Total time: ", time.time() - start, "seconds")
print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))

# --- Cell 25 ---
df=pd.DataFrame({'Train_IoU' : history.history['iou_score'],'Val_IoU' : history.history['val_iou_score'] }, columns=['Train_IoU','Val_IoU'])
EXCEL_DIR = r"C:\Users\User\Desktop\tesis\callbacks"
os.makedirs(EXCEL_DIR, exist_ok=True)
df.to_excel(os.path.join(EXCEL_DIR, MODEL_NAME + '.xlsx'))

# --- Cell 26 ---
ids = np.arange(len(test_dataset))

for i in ids:

    image, gt_mask = test_dataset[i]
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image).round()

    visualize(
        image=denormalize(image.squeeze()),
        gt_mask=gt_mask[..., 0].squeeze(),
        pr_mask=pr_mask[..., 0].squeeze(),
    )


