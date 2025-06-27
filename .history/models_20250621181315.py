# models.py  (sólo cambian las arquitecturas)
import segmentation_models as sm
import tensorflow as tf

from tensorflow.keras.metrics import Precision, Recall
from tensorflow import keras
from deeplabv3p.model import Deeplabv3 as DeepLabV3Plus

def build_model(arch, backbone, n_classes, activation, lr, input_shape=(384,384,3), dropout=0.3,freeze_encoder=False):# eliminar freezer
    if arch == 'baseline':
        base = sm.Unet(backbone, classes=n_classes, activation=None, encoder_weights='imagenet', encoder_freeze=freeze_encoder,input_shape=input_shape)
    elif arch == 'pspnet':
        base = sm.PSPNet(backbone, classes=n_classes, activation=None, encoder_weights='imagenet', encoder_freeze=freeze_encoder,input_shape=input_shape)
    elif arch == 'fpn':
        base = sm.FPN(backbone, classes=n_classes, activation=None, encoder_weights='imagenet', encoder_freeze=freeze_encoder,input_shape=input_shape)
        #base = DeepLabV3Plus(backbone,input_shape=input_shape,classes=n_classes,activation=None,eights='imagenet')  # o None para entrenar desde cero)
    elif arch == 'deeplabv3+':
        base = DeepLabV3Plus(
            weights='imagenet,
            input_shape=input_shape,
            classes=n_classes,
            backbone=backbone,
            activation=None      # <-- typo y valor inválido
        )

    
    else:
        raise ValueError(f"Arquitectura desconocida: {arch}")

    x = keras.layers.Dropout(dropout)(base.output)
    x = keras.layers.Activation(activation, name=activation)(x)
    model = keras.models.Model(inputs=base.input, outputs=x)

    optim   = keras.optimizers.Adam(lr)
    loss    = sm.losses.DiceLoss()
    metrics = [
        sm.metrics.IOUScore(threshold=0.5),  # IoU
        Precision(name='precision'),         # Precisión
        Recall(name='recall'),               # Recall
        sm.metrics.FScore(threshold=0.5)     # F1-Score
    ]
    model.compile(optimizer=optim, loss=loss, metrics=metrics)
    return model
