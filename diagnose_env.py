#!/usr/bin/env python3
"""
diagnose_env.py

Recolección de información de diagnóstico para tu experimento.
Genera un archivo "env_diagnostic.txt" con:
 - Versiones de librerías clave
 - Información de CUDA y GPU
 - Uso de memoria actual
 - Salida de nvidia-smi
 - Variables de entorno CUDA relevantes
"""

import torch
import torchvision
import kornia
import albumentations
import segmentation_models_pytorch as smp
import six
import os
import subprocess

def run_command(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode()
    except Exception as e:
        return f"Error al ejecutar '{cmd}': {e}"

def main():
    lines = []
    lines.append("=== DIAGNÓSTICO DE ENTORNO ===\n")
    lines.append("**Versiones de Python y librerías**")
    lines.append(f"Python            : {os.sys.version.replace(os.linesep,' ')}")
    lines.append(f"PyTorch           : {torch.__version__}")
    lines.append(f"CUDA (torch)      : {torch.version.cuda}")
    lines.append(f"TorchVision       : {torchvision.__version__}")
    lines.append(f"Kornia            : {kornia.__version__}")
    lines.append(f"Albumentations    : {albumentations.__version__}")
    lines.append(f"SMP (smp.__version__): {smp.__version__}")
    lines.append(f"six               : {six.__version__}\n")

    lines.append("**GPU disponible**")
    lines.append(f"cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"Device count      : {torch.cuda.device_count()}")
        lines.append(f"Device name       : {torch.cuda.get_device_name(0)}")
        lines.append(f"Current device    : {torch.cuda.current_device()}")
        lines.append(f"Memory allocated  : {torch.cuda.memory_allocated()/(1024**3):.2f} GB")
        lines.append(f"Memory reserved   : {torch.cuda.memory_reserved()/(1024**3):.2f} GB\n")

    lines.append("**nvidia-smi**")
    lines.append(run_command("nvidia-smi"))

    lines.append("\n**Variables de entorno CUDA**")
    for var in ["CUDA_VISIBLE_DEVICES","CUDA_LAUNCH_BLOCKING","PYTORCH_CUDA_ALLOC_CONF"]:
        lines.append(f"{var}={os.environ.get(var,'')}")

    out_path = "env_diagnostic.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Diagnóstico guardado en {out_path}")

if __name__ == "__main__":
    main()
