# losses.py
import torch
import torch.nn as nn
import kornia.morphology as kornia_morph
import segmentation_models_pytorch as smp   # ⇦ para usar DiceLoss ya implementado

class LossConsistenciaMorfologicaCompuesta(nn.Module):
    """
    Calcula la pérdida de consistencia entre la predicción del *student*
    y dos objetivos morfológicos generados a partir de la salida del *teacher*:
        • dilatación(teacher)   ≈ bordes externos
        • erosión(teacher)      ≈ bordes internos
    Para cada kernel elíptico definido por el usuario se suman las dos pérdidas Dice;
    luego se promedia sobre todos los kernels.
    """
    def __init__(self, lista_kernels):
        super().__init__()
        self.kernels = lista_kernels
        # Usamos la implementación oficial de SMP; como pasaremos PROBABILIDADES
        # (torch.sigmoid student/teacher) fijamos from_logits=False.
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=False)

    @torch.cuda.amp.autocast(enabled=False)  # sin mezcla de precisión dentro
    def forward(self, pred_student_prob, teacher_prob):
        """
        Parámetros
        ----------
        pred_student_prob : Tensor [B,1,H,W] – salida sigmoid del student
        teacher_prob      : Tensor [B,1,H,W] – salida sigmoid del teacher (sin grad)
        """
        teacher_prob = teacher_prob.detach()          # no back-prop al teacher
        loss_total   = 0.0

        for kernel in self.kernels:
            # Kornia recibe el kernel como Tensor [1,H,W] o [B,1,H,W]
            # Generamos los objetivos morfológicos
            obj_dil = kornia_morph.dilation( teacher_prob, kernel)
            obj_ero = kornia_morph.erosion( teacher_prob, kernel)

            # Dice(student , objetivo)  para dilatación y erosión
            loss_kernel = self.dice(pred_student_prob, obj_dil) + \
                          self.dice(pred_student_prob, obj_ero)

            loss_total += loss_kernel
            # Liberar cualquier reserva extra de memoria en GPU
            torch.cuda.empty_cache()

        # Promediamos sobre todos los kernels
        return loss_total / len(self.kernels)
