import torch
import cv2
import numpy as np

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