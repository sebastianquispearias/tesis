import torch
import cv2
import numpy as np

#
# utils.py
import torch
import subprocess
import logging

def gpu_snapshot(note: str = None):
    """Captura nvidia-smi y lo manda a logging.info (solo llama a subprocess)."""
    try:
        smi = subprocess.check_output(["nvidia-smi"], text=True)
        header = f"\n[nvidia-smi snapshot{f' {note}' if note else ''}]\n"
        logging.info(header + smi)
    except Exception:
        logging.warning("No se pudo ejecutar nvidia-smi")

def reset_memory(device):
    """Resetea las estadísticas de pico de memoria."""
    torch.cuda.reset_peak_memory_stats(device)

def log_memory_epoch(device, epoch):
    """Loggea alloc / reserved / peak_reserved al final de la época."""
    alloc    = torch.cuda.memory_allocated(device)    / 1024**3
    reserved = torch.cuda.memory_reserved(device)     / 1024**3
    peak     = torch.cuda.max_memory_reserved(device) / 1024**3
    #logging.info(
    #    f"[MEM][Epoch {epoch}] alloc={alloc:.2f} GiB  "
    #    f"reserved={reserved:.2f} GiB  peak_reserved={peak:.2f} GiB"
    #)
    return alloc, reserved, peak

def summarize_epochs(stats: dict, epochs: int):
    """
    Imprime una tabla compacta con todas las épocas y métricas requeridas.
    """
    header = (
        "Epoch |  Sup   |  Cons  | Total  | Alloc  | Reserved | Peak   | "
        "Val_loss |  IoU   |   P    |   R    |  F1   "
    )
    sep = "-" * len(header)
    logging.info("\n" + header)
    logging.info(sep)
    for e in range(epochs):
        logging.info(
            f"{e+1:5d} | "
            f"{stats['sup'][e]:6.4f} | {stats['cons'][e]:6.4f} | {stats['total'][e]:7.4f} | "
            f"{stats['alloc'][e]:6.2f} | {stats['reserved'][e]:8.2f} | {stats['peak'][e]:7.2f} | "
            f"{stats['val_loss'][e]:8.4f} | {stats['iou'][e]:7.4f} | "
            f"{stats['p'][e]:7.4f} | {stats['r'][e]:7.4f} | {stats['f1'][e]:7.4f}"
        )

#
def crear_kernel_elipse(diametro_equiv: int, aspect_ratio: float, angulo_grados: int, device: str = 'cuda'):
    """
    Crea un kernel morfológico con forma de elipse y una orientación específica.
    """
    # Calcular radios de los ejes a partir del diámetro y aspect ratio
    radio_menor = (diametro_equiv / 2.0) / np.sqrt(aspect_ratio)
    radio_mayor = radio_menor * aspect_ratio
    
    # Tamaño del lienzo para dibujar la elipse
    tam_lienzo = int(radio_mayor * 2.5)
    if tam_lienzo % 2 == 0: tam_lienzo += 1 # Asegurar tamaño impar
    centro = (tam_lienzo // 2, tam_lienzo // 2)

    # Crear elipse
    eje = (int(round(radio_menor)), int(round(radio_mayor)))
    lienzo = np.zeros((tam_lienzo, tam_lienzo), dtype=np.uint8)
    cv2.ellipse(lienzo, center=centro, axes=eje, angle=angulo_grados, startAngle=0, endAngle=360, color=255, thickness=-1)
    
    kernel = (lienzo > 0).astype(np.float32)
    
    return torch.from_numpy(kernel).to(device)

@torch.no_grad()
def actualizar_teacher_ema(student_model, teacher_model, alpha=0.999):
    for param_teacher, param_student in zip(teacher_model.parameters(), student_model.parameters()):
        param_teacher.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)