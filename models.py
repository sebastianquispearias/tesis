# models.py  (sólo cambian las arquitecturas)
import segmentation_models as sm
from tensorflow import keras

def build_model(arch, backbone, n_classes, activation, lr, input_shape=(320,320,3), dropout=0.3):
    if arch == 'baseline':
        base = sm.Unet(backbone, classes=n_classes, activation=None, encoder_weights='imagenet',input_shape=input_shape)
    elif arch == 'pspnet':
        base = sm.PSPNet(backbone, classes=n_classes, activation=None, encoder_weights='imagenet',input_shape=input_shape)
    elif arch == 'fpn':
        base = sm.FPN(backbone, classes=n_classes, activation=None, encoder_weights='imagenet',input_shape=input_shape)
    else:
        raise ValueError(f"Arquitectura desconocida: {arch}")

    x = keras.layers.Dropout(dropout)(base.output)
    x = keras.layers.Activation(activation, name=activation)(x)
    model = keras.models.Model(inputs=base.input, outputs=x)

    optim   = keras.optimizers.Adam(lr)
    loss    = sm.losses.DiceLoss()
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    model.compile(optimizer=optim, loss=loss, metrics=metrics)
    return model
