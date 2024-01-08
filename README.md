# Software_Defect_Prediction

## Ensemble Learning for Software Defect Prediction using Robust Feature Selection
This repository contains code and data for my project on applying ensemble learning with robust feature selection to enhance software defect prediction

## Abstract
Software defects can substantially degrade system quality and reliability. Identifying fault-prone components early in development is crucial for effective quality assurance. This research investigates a robust feature selection-based ensemble learning framework for enhanced software defect prediction. The approach leverages Robust PCA to construct an informative low-dimensional feature set from software metrics which retains predictive signal while reducing noise. The filtered features are used to train an integrated ensemble combining bagging, boosting, and stacking for diversity.

Evaluation on five public NASA datasets indicates significant gains over both single and ensemble classifiers. The heterogeneous framework achieved a top ROC AUC of 0.59 and reduced imbalance via improved precision and recall. The results demonstrate the benefits of coupling robust transform-based feature engineering with deliberate ensemble design for advancing defect prediction capabilities.

## Methodology
The proposed defect prediction framework consists of:

Dataset Selection: 5 public NASA promise datasets - JM1, CM1, KC1, KC2, PC1
Feature Selection: Apply Robust PCA for dimensionality reduction
Classification: Integrated ensemble with bagging, boosting and stacking
Evaluation: Metrics - accuracy, AUC, precision, recall, F1-score

## Contents
Datasets (not included due to size)
Jupyter notebooks
Exploratory Data Analysis
Classic PCA Analysis
Robust PCA Analysis
Ensemble Framework
