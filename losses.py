import torch
import torch.nn as nn
import kornia.morphology as kornia_morph
import torch.nn.functional as F

class LossConsistenciaMorfologicaCompuesta(nn.Module):
    def __init__(self, lista_kernels):
        super().__init__()
        self.kernels = lista_kernels
        self.comparador_loss = nn.MSELoss()

    def forward(self, prediccion_student, prediccion_profesor):
        pred_profesor_sin_grad = prediccion_profesor.detach()
        loss_total = 0.0

        for kernel in self.kernels:
            # calculamos tamaño y padding del kernel
            K = kernel.shape[-1]
            pad = K // 2

            # Dilatación ≈ max-pool2d
            objetivo_dilatado = F.max_pool2d(
                pred_profesor_sin_grad, kernel_size=K, stride=1, padding=pad
            )
            # Erosión ≈ -max-pool2d(-x)
            objetivo_erosionado = -F.max_pool2d(
                -pred_profesor_sin_grad, kernel_size=K, stride=1, padding=pad
            )
            loss_kernel = self.comparador_loss(prediccion_student, objetivo_erosionado) + \
                          self.comparador_loss(prediccion_student, objetivo_dilatado)
            
            loss_total += loss_kernel
            torch.cuda.empty_cache()            
        # Devolvemos el promedio de los losses de todos los kernels
        return loss_total / len(self.kernels)