# KNN from Scratch on Wine Dataset 🍇

This project implements the k-Nearest Neighbors (k-NN) algorithm from scratch using Python and NumPy.  
We evaluate the classifier on the UCI Wine dataset, explore the impact of different distance metrics (Euclidean, Manhattan, Minkowski), and analyze performance with various K values.

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

---

## 🚀 How to Run the Project

Follow the steps below to run the custom k-NN classifier and reproduce the experiments:

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/wine-knn-project.git
cd wine-knn-project
```

### 2️⃣ Install dependencies

install manually:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3️⃣ Make sure the dataset file is in place

Ensure that `wine.data` is present in the root directory of the project.  
You can download it from the [UCI Wine Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data).

### 4️⃣ Run the analysis

You can either:
- Open `analysis.ipynb` in Jupyter Notebook and run the cells step by step  
**or**
- Execute the Python script if you convert it:
```bash
python analysis.py
```

---

## 🧪 Testing the Classifier Manually

You can also test the custom KNN class in any script:

```python
from knn import KNN

model = KNN(k=5, distance_metric="euclidean")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
