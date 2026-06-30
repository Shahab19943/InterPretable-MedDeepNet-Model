# Interpretable-MedDeepNet-Model
This repository contains the implementation of interpretable MedDeepNet for lung CT scan classification and explainable AI-based interpretation. The framework integrates deep learning, feature selection, and LIME-based explanation techniques for medical imaging analysis.

## Key Features
- Deep CNN model (MedDeepNet) for lung CT classification
- preprocessing pipeline
- Feature extraction using a trained MedDeepNet
- Atom Search Optimization (ASO) for feature selection
- Machine learning-based classification (SVM, KNN, etc.)
- LIME-based interpretable model explanations
- Performance metrics for explanation evaluation

## Dataset
The dataset consists of lung CT scan images with three classes:
- Benign
- Malignant
- Normal

Dataset path is not included in this repository due to size constraints.
Users should organize the dataset as:

dataset/
   original/
   resized_augmented/
   segmented/


## How to Run

Run scripts in the following order:

1. preprocess_pipeline.m
2. train_meddeepnet.m
3. feature_extraction.m
4. feature_selection.m
5. classification.m
6. lime_base_explanation.m


## Requirements
- MATLAB R2022b or later
- Deep Learning Toolbox
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

- ## Citation
If you use this code, please cite the related paper:

@article{MedDeepNet2026,
  title={Interpretable MedDeepNet: Deep Feature Learning with Atom Search Optimization for Explainable Lung Cancer Detection in CT Images},
  author={Shahab Ul Hassan et al.},
  year={2026}
}
