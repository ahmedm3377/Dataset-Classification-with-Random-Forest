from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

print("The data shape is: ", X.shape)
classes = np.unique(y)
print("Number of classes = {}".format(len(classes)))

# Visualize ranges of each class

feature_names = iris.feature_names
class_names = iris.target_names

# Create a 1x4 subplot: each boxplot corresponds to one feature
fig, axes = plt.subplots(1, 4, figsize=(20, 10))

for i in range(4):  # Loop through features
    data = [X[y == cls, i] for cls in np.unique(y)]  # Group by class
    axes[i].boxplot(data, labels=class_names)
    axes[i].set_title(feature_names[i])
    axes[i].set_ylabel("Value")
    axes[i].set_xlabel("Class")

plt.tight_layout()
plt.show()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier for multiclass classification
clf_multiclass = RandomForestClassifier(n_estimators=10)

# Train the model
clf_multiclass.fit(X_train, y_train)

# Make predictions
predictions_multiclass = clf_multiclass.predict(X_test)

# Evaluate metrics for multiclass classification
accuracy_multiclass = accuracy_score(y_test, predictions_multiclass)
print("Multiclass Classification Accuracy: {}".format(accuracy_multiclass))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions_multiclass))
print("Classification report:")
print(classification_report(y_test, predictions_multiclass))