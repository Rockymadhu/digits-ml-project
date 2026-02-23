# Supervised and Unsupervised Learning using Digits Dataset

## Project Overview
This project demonstrates:
- Supervised Learning using K-Nearest Neighbors (KNN)
- Unsupervised Learning using K-Means clustering
- Model evaluation using accuracy, confusion matrix, and classification report
- PCA visualization of clusters

## Dataset
- Digits dataset from Scikit-learn
- 1797 handwritten digit images
- 64 features per image
- 10 classes (0–9)

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Run:
python main.py

## Output
Dataset shape: (1797, 64)
Number of classes: 10
Data preparation completed.

KNN Model Training Completed.

Accuracy: 0.9694444444444444

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        33
           1       0.93      1.00      0.97        28
           2       0.94      0.97      0.96        33
           3       0.97      0.97      0.97        34
           4       0.98      1.00      0.99        46
           5       0.98      0.96      0.97        47
           6       0.97      1.00      0.99        35
           7       1.00      0.97      0.99        34
           8       0.97      0.93      0.95        30
           9       0.95      0.90      0.92        40

    accuracy                           0.97       360
   macro avg       0.97      0.97      0.97       360
weighted avg       0.97      0.97      0.97       360


K-Means Clustering Completed.

Project Completed Successfully.
