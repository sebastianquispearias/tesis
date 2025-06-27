import torch
import torch.nn as nn
import kornia.morphology as kornia_morph

class LossConsistenciaMorfologicaCompuesta(nn.Module):
    def __init__(self, lista_kernels):
        super().__init__()
        self.kernels = lista_kernels
        self.comparador_loss = nn.MSELoss()

    def forward(self, prediccion_student, prediccion_profesor):
        pred_profesor_sin_grad = prediccion_profesor.detach()
        loss_total = 0.0

        # Itera sobre la librería de kernels y acumula el loss
        for kernel in self.kernels:
            objetivo_erosionado = kornia_morph.erosion(pred_profesor_sin_grad, kernel)
            objetivo_dilatado = kornia_morph.dilation(pred_profesor_sin_grad, kernel)
            
            loss_kernel = self.comparador_loss(prediccion_student, objetivo_erosionado) + \
                          self.comparador_loss(prediccion_student, objetivo_dilatado)
            
            loss_total += loss_kernel
        
        # Devolvemos el promedio de los losses de todos los kernels
        return loss_total / len(self.kernels)