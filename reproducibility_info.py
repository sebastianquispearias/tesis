#!/usr/bin/env python
import sys, os, subprocess
import tensorflow as tf
import segmentation_models as sm
import cv2

def main():
    print("=== VERSIÓN ENTORNO ===")
    print("Python      :", sys.version.replace("\n",""))
    print("OS          :", sys.platform, os.name)
    print("TensorFlow  :", tf.__version__)
    print("Built w/CUDA:", tf.test.is_built_with_cuda())
    build_info = tf.sysconfig.get_build_info()
    print("CUDA        :", build_info.get('cuda_version'))
    print("cuDNN       :", build_info.get('cudnn_version'))
    print("GPUs        :", tf.config.list_physical_devices('GPU'))
    print("segmentation_models:", sm.__version__)
    print("OpenCV      :", cv2.__version__)
    print("\n=== PAQUETES PIP FREEZE ===")
    subprocess.run([sys.executable, "-m", "pip", "freeze"])
    
if __name__ == "__main__":
    main()
