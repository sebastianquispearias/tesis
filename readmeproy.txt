README – Corrosion Segmentation Experiments
(VT Bridge + Ameli Datasets, Binary Masks)

1 · Definición del problema
Detectar corrosión en imágenes de inspección de puentes mediante segmentación semántica binaria.
El objetivo es cuantificar la capacidad de modelos CNN modernos para generalizar con escasez de etiquetas y capitalizar datos sin coste de anotación (unlabeled) procedentes del mismo dominio o de dominios cercanos.

2 · Datasets y conteos finales
Conjunto	Fuente	Etiquetas	Resolución	# imágenes
VT-Bridge	Virginia Tech Bridge Inspection Reports	Sí	512² → 384²	440
Ameli	[Hebdon & Bianchi, UT Austin]	No (solo imágenes)	512² → 384²	514
Total procesado	–	–	384² (PNG)	954

2.1 Pre-procesamiento
Resize → 384 × 384 px.

Máscaras multiclase → binaria:

pixel = 0 ⇒ fondo

pixel > 0 ⇒ corrosión (255)

Carpetas generadas:

kotlin
Copiar
Editar
data/
 └─ processed_binary_384/
     ├─ images/
     └─ masks/
2.2 Splits supervisados
70 / 15 / 15 % estratificado sobre VT (440 img)

Split	# imágenes
vt_train_total	308
vt_val	66
vt_test	66

Sub-porcentajes para escasez de etiquetas
Regímenes	10 %	25 %	50 %
vt_train_X	31	77	154

Los archivos .txt con rutas se encuentran en splits/.

3 · Augmentations
css
Copiar
Editar
Resize 384² → HorizontalFlip → VerticalFlip
→ RandomRotate90 → Transpose → GridDistortion
(Albumentations 1.4.3) – aplicadas sólo a train.

4 · Diseño experimental
Código	Descripción
A-Full	100 % etiquetas VT (308)
B-10/25/50	10 / 25 / 50 % etiquetas VT; resto ignorado
C-10/25/50	10 / 25 / 50 % etiquetas VT labeled + 90 / 75 / 50 % VT unlabeled + 100 % Ameli unlabeled
(FixMatch + EMA)
D-Ablation	Comparación de cabezas (U-Net, FPN, PSPNet) con 25 % etiquetas

Ameli se usa solo como unlabeled para pseudo-etiquetado; las métricas se calculan exclusivamente en vt_val y vt_test.

5 · Modelos y parámetros
Parámetro	Valor
Architecturas	baseline = U-Net + EfficientNet-B3, fpn, pspnet
Entrada	3 × 384 × 384
Pérdida	BinaryCrossentropy + Dice
Optimizador	Adam
LR inicial	1 × 10⁻⁴
Scheduler	ReduceLROnPlateau (patience = 5, factor = 0.5)
Batch	4
Epochs	80 (sin early-stopping)
Seeds (5-fold)	42, 77, 1337, 2025, 31415
Métricas	IoU (primary), Dice, Precision, Recall

Validación cruzada 5-fold: estratificada sobre vt_train_total; se reporta media ± σ.

6 · Hardware de referencia
Recurso	Valor
GPU	NVIDIA RTX 3060 12 GB
RAM	64 GB
SO	Ubuntu 22.04 / CUDA 12.3
Tiempo medio/época	93 s (VT-25 %, batch = 4)

7 · Estructura del repositorio
kotlin
Copiar
Editar
project_root/
 ├─ data/
 │   └─ processed_binary_384/
 ├─ splits/
 │   ├─ vt_train.txt
 │   ├─ vt_train_10.txt
 │   ├─ vt_train_25.txt
 │   └─ vt_train_50.txt
 ├─ scripts/
 │   ├─ prepare_dataset_final.py
 │   ├─ make_subsplits.py
 │   └─ train.py
 ├─ models/
 ├─ callbacks_monitor.py
 └─ README.md  ← (este archivo)
8 · Reproducción paso a paso
bash
Copiar
Editar
# 1. Crear entorno
conda create -n corroseg python=3.10
conda activate corroseg
pip install -r requirements.txt

# 2. Preparar dataset
python scripts/prepare_dataset_final.py \
  --src_vt raw/VT --src_ameli raw/Ameli --out data/processed_binary_384

# 3. Generar sub-splits (ya provistos – repita si quiere otra seed)
python scripts/make_subsplits.py \
  --train_csv splits/vt_train.txt --perc 0.25 --out splits/vt_train_25.txt --seed 42

# 4. Entrenar (ejemplo: PSPNet, régimen C-25, fold-seed 42)
python scripts/train.py \
  --arch pspnet --regime C25 --seed 42 \
  --train_list splits/vt_train_25.txt --val_list splits/vt_val.txt

# 5. Evaluar
python scripts/evaluate.py --model_ckpt runs/pspnet_C25_seed42/best.h5 --test_list splits/vt_test.txt
9 · Checklist de reproducibilidad (ML-Reproducibility Checklist 2023)
 Código, datos y scripts para splits incluidos

 Hiper-parámetros explícitos

 Seeds fijadas y listadas

 Métricas reportadas con varianza

 Requisitos de hardware y tiempo por época documentados

 Procedimiento completo de entrenamiento y evaluación descrito

10 · Licencia y atribuciones
Los datos de VT-Bridge y Ameli se distribuyen bajo las licencias académicas indicadas por sus autores originales.
El código y la documentación de este repositorio se publican bajo licencia MIT.

