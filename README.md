![image](https://github.com/user-attachments/assets/aae8d15a-d0ed-462b-b06a-57ad24c93340)# Semi-Supervised Corrosion Segmentation with Adaptive Morphological Regularization

## Project Summary

This repository contains the PyTorch implementation of a novel semi-supervised learning (SSL) framework designed for high-precision segmentation of corrosion defects in scenarios with a scarcity of labeled data.

The main contribution is a new **morphological consistency loss** integrated into the Mean Teacher framework. Unlike classic methods that enforce consistency at the pixel level, our method compels the student model to generate predictions that are structurally coherent with the teacher's morphological pseudo-targets (erosion and dilation).

The second layer of innovation is that the **morphological kernels are domain-adaptive**: they are designed based on a prior statistical analysis of the shape, size, and orientation of real defects, creating a much more powerful and informed shape regularizer than generic approaches.


## Key Features

* **Framework**: PyTorch
* **Main Model**: DeepLabV3+ with an EfficientNet-B3 backbone.
* **Implemented Methods**:
    * Supervised Training (for Bxx baselines).
    * Classic Mean Teacher (for Cxx baselines).
    * Our proposed method with adaptive morphological regularization (Dxx).
* **Novel Loss**: `LossConsistenciaMorfologicaCompuesta` which operates with a library of kernels and a flexible internal comparator (MSE or Dice).
* **Domain Intelligence**: Scripts to statistically analyze masks and generate custom elliptical kernels.
* 
Weights & Logs
Due to their size, the pre-trained model weights and the training logs for all experiments are not included in this repository. They can be downloaded from the following Google Drive folder:
https://drive.google.com/drive/folders/1nvSOg8QUZ_HMz7McE3q0UN1Icb_oi518

## Repository Structure

```text
.
├── data/                     
│   └── all_results/
│       └── regimes/
│           ├── B25/
│           ├── C25/
│           └── ...
├── main_supervised_pytorch.py       # SCRIPT 1: Train supervised baselines (Bxx)
├── main_semisup_pytorch.py          # SCRIPT 2: Train SSL method (Cxx) 
├── main_semisup_pytorch_enhancer.py # SCRIPT 3: Train SSL method (Dxx)
├── inferencia_pytorch.py            # SCRIPT 4: Evaluate a model on an image
├── analizar_morfologia_mascaras.py  # TOOL 1: Analyze masks
├── utils.py                         # Helper functions (kernel creation, EMA)
├── losses.py                        # Definition of the composite morphological loss
└── README.md                        # This file
```
Environment Setup
Using conda to manage the environment is recommended.

1.Create the environment:

```text
conda create --name corrosion_semisup python=3.8
conda activate corrosion_semisup
```
2.Install dependencies
```text
# Install PyTorch for your CUDA version (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
# Install the rest of the libraries
pip install segmentation-models-pytorch albumentations kornia scikit-image pandas tqdm opencv
```
Workflow and Usage
The complete process from data to prediction is as follows:

Step 1: Statistical Analysis (Perform only once)
Analyze your labeled training masks to obtain shape statistics.
```text
python analizar_morfologia_mascaras.py "path/to/your/labeled_masks"
```
Note down the results (P25/P75 diameters, aspect ratio, etc.) to use them in the training scripts.

Step 2: Model Training
Edit the training scripts to ensure the hyperparameters (UNLABELED_W, cons_loss type, etc.) are correct for the experiment you want to run.

For Supervised Baselines (Bxx):
```text
python main_supervised_pytorch.py --regime B25
```
For Semi-Supervised Baselines (Cxx and Dxx):
```text
python main_semisup_pytorch.py --regime C25
python main_semisup_pytorch_enhancer.py --regime C25
```
Step 3: Inference and Visualization
Use the inference script to evaluate a saved model on a specific image.
```text
python inferencia_pytorch.py --weights best_model_D25.pth --input "path/to/an/image.png" --mask "path/to/the/mask.png"
```
Expected Results
The goal of this research is to complete the following comparative table, demonstrating that the Dxx method (our proposal) outperforms the Bxx and Cxx baselines.
```text
% of Labeled Data     Supervised (Bxx)    Classic MT (Cxx)    Morphological Regularization (Dxx)

25%                      0.5305              0.6195                     0.6186

50%                      0.5186              0.6020                     0.6088

75%                      0.4890              0.5876                     0.5612
```

