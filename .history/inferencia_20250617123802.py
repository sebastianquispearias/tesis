#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de inferencia para el modelo de segmentación de corrosión.
Permite cargar pesos entrenados, procesar una imagen, calcular IoU opcionalmente y visualizar resultados.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import segmentation_models as sm

from models import build_model


def parse_args():
    p = argparse.ArgumentParser(
        description='Inferencia de segmentación de corrosión'
    )
    p.add_argument('--arch',
                   choices=['baseline', 'pspnet', 'fpn', 'deeplabv3+'],
                   default='baseline',
                   help='Arquitectura del modelo')
    p.add_argument('--backbone',
                   default='efficientnetb3',
                   help='Backbone para la arquitectura')
    p.add_argument('--weights',
                   required=True,
                   help='Ruta al archivo de pesos (.h5)')
    p.add_argument('--input',
                   required=True,
                   help='Imagen de entrada para inferencia')
    p.add_argument('--mask',
                   help='Máscara de ground truth para calcular IoU (opcional)')
    p.add_argument('--threshold',
                   type=float,
                   default=0.5,
                   help='Umbral para binarizar la predicción')
    p.add_argument('--no-show',
                   action='store_true',
                   help='No mostrar la visualización')
    return p.parse_args()


def main():
    args = parse_args()

    # Parámetros fijos según tus experimentos
    CLASSES     = ['corrosion']
    LR          = 1e-4
    INPUT_SHAPE = (384, 384, 3)
    n_classes   = 1 if len(CLASSES) == 1 else len(CLASSES) + 1
    activation  = 'sigmoid' if n_classes == 1 else 'softmax'

    # 1) Construir y cargar modelo
    model = build_model(
        args.arch,
        args.backbone,
        n_classes,
        activation,
        LR,
        input_shape=INPUT_SHAPE
    )
    model.load_weights(args.weights)
    print(f"[INFO] Pesos cargados desde: {args.weights}")

    # 2) Preprocesar imagen de entrada
    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        raise FileNotFoundError(f"No se encontró la imagen: {args.input}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, INPUT_SHAPE[:2], interpolation=cv2.INTER_LINEAR)
    preprocess_fn = sm.get_preprocessing(args.backbone)
    inp = preprocess_fn(img_resized)

    # 3) Inferencia
    pred = model.predict(np.expand_dims(inp, 0))[0]
    pred_mask = (pred[..., 0] > args.threshold)
    print(f"[INFO] Predicción realizada con umbral {args.threshold}")

    # 4) Calcular IoU si hay máscara de referencia
    if args.mask:
        gt_bgr = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if gt_bgr is None:
            raise FileNotFoundError(f"No se encontró la máscara: {args.mask}")
        gt_mask = gt_bgr > 0
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        iou = intersection / union if union > 0 else 0.0
        print(f"[RESULT] IoU: {iou:.3f}")

    # 5) Visualización
    if not args.no_show:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img_rgb)
        plt.title('Imagen de entrada')
        plt.axis('off')

        if args.mask:
            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(pred_mask, cmap='gray')
            plt.title('Predicción')
            plt.axis('off')
        else:
            plt.imshow(pred_mask, cmap='gray')
            plt.title('Predicción')
            plt.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
