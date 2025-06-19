import tensorflow as tf
import segmentation_models as sm

# Definir la función de pérdida
dice_loss = sm.losses.DiceLoss()

# Entradas de prueba en float16
y_true_fp16 = tf.ones((1, 64, 64, 1), dtype=tf.float16)
y_pred_fp16 = tf.ones((1, 64, 64, 1), dtype=tf.float16)

# Calcular la pérdida
loss_fp16 = dice_loss(y_true_fp16, y_pred_fp16)
print("DiceLoss with float16 inputs dtype:", loss_fp16.dtype)

# Entradas de prueba en float32
y_true_fp32 = tf.ones((1, 64, 64, 1), dtype=tf.float32)
y_pred_fp32 = tf.ones((1, 64, 64, 1), dtype=tf.float32)

loss_fp32 = dice_loss(y_true_fp32, y_pred_fp32)
print("DiceLoss with float32 inputs dtype:", loss_fp32.dtype)
