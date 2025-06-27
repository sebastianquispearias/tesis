# models.py  (sólo cambian las arquitecturas)
import segmentation_models as sm
import tensorflow as tf

from tensorflow.keras.metrics import Precision, Recall
from tensorflow import keras
from deeplabv3p.model import Deeplabv3 as DeepLabV3Plus

def build_model(
    arch,
    backbone,       # aquí no se usa: el exp de Pascal VOC viene predefinido con Xception
    n_classes,
    activation,
    lr,
    input_shape=(384, 384, 3),
    freeze_encoder=False
):
    if arch == 'deeplabv3+':
        # 1) Carga la config de DeepLabV3+ | Pascal VOC
        exp_config = get_exp_config('seg_deeplabv3plus_pascal')

        # 2) Crea la Task (monta modelo + lógica interna)
        task = get_task(exp_config.task, logging_dir=None)

        # 3) Construye el modelo base (tf.keras.Model)
        base = task.build_model()

        # 4) Injerta tu propia cabeza de salida para 'n_classes'
        x = base.output
        x = tf.keras.layers.Conv2D(
            filters=n_classes,
            kernel_size=1,
            activation=activation,
            name='custom_logits'
        )(x)
        model = tf.keras.Model(inputs=base.input, outputs=x)

        # 5) (Opcional) congelar encoder
        if freeze_encoder:
            for layer in model.layers:
                # en la config original el encoder es Xception; 
                # si quisieras congelar, podrías usar layer.name.startswith('entry') etc.
                if layer.name.startswith('entry') or layer.name.startswith('middle') or layer.name.startswith('exit'):
                    layer.trainable = False

        # 6) Compila con tu optimizador y tus pérdidas/métricas
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.BinaryCrossentropy() if n_classes == 1 else 'categorical_crossentropy',
            metrics=[
                # importa y usa tus métricas definidas en metrics_utils
                # e.g. iou_score, precision, recall, f1_score
            ]
        )
        return model
