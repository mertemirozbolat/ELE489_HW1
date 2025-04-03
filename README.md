# KNN from Scratch on Wine Dataset 🍷

This project implements the k-Nearest Neighbors (k-NN) algorithm from scratch using Python and NumPy.  
I evaluate the classifier on the UCI Wine dataset, explore the impact of different distance metrics (Euclidean, Manhattan, Minkowski), and analyze performance with various K values.

## 🔧 Features
- Custom-built k-NN (no sklearn)
- Euclidean, Manhattan, Minkowski distance support
- Accuracy vs. K plot
- Confusion Matrix and Classification Report (visualized)

## 📊 Summary of Findings
- Euclidean distance achieved highest average accuracy
- K=5 generally offered the best balance between bias and variance
- Visual inspection of confusion matrices shows strong class separation

## 📁 Files
- `knn.py` – The main KNN class
- `analysis.ipynb` – All experiments, plots, and evaluation
- `README.md` – Project documentation

## 📚 Dataset
- UCI Wine Dataset: [Link](https://archive.ics.uci.edu/ml/datasets/wine)
