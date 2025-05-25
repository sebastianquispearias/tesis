import tensorflow as tf
from tensorflow import keras

class ProgressMonitor(keras.callbacks.Callback):
    def __init__(self, batch_print_freq=50):
        super().__init__()
        self.batch_print_freq = batch_print_freq

    def _get_mem_info(self):
        # devuelve {'current':…, 'peak':…} o None
        if hasattr(tf.config.experimental, 'get_memory_info'):
            return tf.config.experimental.get_memory_info('GPU:0')
        return None

    def on_train_begin(self, logs=None):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f" SÍ-- GPU detectada: {gpus[0].name}")
        else:
            print(" NO-- no se detectó GPU, usaré CPU")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n--- Inicio de época {epoch+1}/{self.params['epochs']} ---")

    def on_train_batch_begin(self, batch, logs=None):
        if batch % self.batch_print_freq == 0:
            info = self._get_mem_info()
            if info:
                cur  = info['current'] // (1024**2)
                peak = info['peak']    // (1024**2)
                print(f"[Batch {batch}] MEM antes → cur={cur} MB, peak={peak} MB")

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.batch_print_freq == 0:
            info = self._get_mem_info()
            if info:
                cur  = info['current'] // (1024**2)
                peak = info['peak']    // (1024**2)
                print(f"[Batch {batch}] MEM después → cur={cur} MB, peak={peak} MB")
            print(f"  Batch {batch}/{self.params['steps']} — loss: {logs.get('loss',0):.4f}")

    def on_epoch_end(self, epoch, logs=None):
        loss  = logs.get('loss',       0)
        iou   = logs.get('iou_score',  0)
        vloss = logs.get('val_loss',   0)
        viou  = logs.get('val_iou_score', 0)
        print(f"Época {epoch+1} → loss:{loss:.4f}, iou:{iou:.4f} | val_loss:{vloss:.4f}, val_iou:{viou:.4f}")
        info = self._get_mem_info()
        if info:
            cur  = info['current'] // (1024**2)
            peak = info['peak']    // (1024**2)
            print(f" GPU mem — current: {cur} MB, peak: {peak} MB")
