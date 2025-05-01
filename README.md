# 🌸 Iris Classification with Random Forest and Feature Visualization

This project demonstrates how to classify Iris flower species using a **Random Forest** classifier and visualize the distribution of each feature across classes using **boxplots**.

## 📊 Dataset

The [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) is a classical multivariate dataset introduced by R.A. Fisher. It consists of **150 samples** from three species of Iris:

- *Iris setosa*
- *Iris versicolor*
- *Iris virginica*

Each sample contains **4 features**:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

## 🧭 Project Overview

### ✅ Features
- 📦 Load and explore the Iris dataset using `scikit-learn`.
- 📉 Visualize feature distributions with **boxplots** for each class.
- 🌲 Train a **Random Forest** classifier for multiclass classification.
- 📈 Evaluate the model using accuracy, confusion matrix, and classification report.

### 📌 Steps

1. **Data Loading**  
   Load the dataset using `load_iris()` and extract features and labels.

2. **Data Visualization**  
   Use `matplotlib` to create a `1x4` subplot showing **boxplots** of each feature across the three Iris classes to understand data distribution.

3. **Train-Test Split**  
   Split the data into training (80%) and test (20%) sets.

4. **Model Training**  
   Train a `RandomForestClassifier` using the training data.

5. **Evaluation**  
   Evaluate the classifier on the test data using:
   - Accuracy score
   - Confusion matrix
   - Classification report (precision, recall, F1-score)

---

## 📷 Example Visualization

Each plot below shows a **boxplot for one feature** across the three classes:

Sepal Length	| Sepal Width	| Petal Length	| Petal Width

Boxplot	      | Boxplot	    | Boxplot	      | Boxplot

These plots help identify which features are most discriminative for classification.

---

## 🧪 Evalution metrics

- Multiclass Classification Accuracy
- Confusion matrix
- Classification report: precision, recall, and f1-score
