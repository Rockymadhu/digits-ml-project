# ===============================
# Supervised & Unsupervised ML
# Using Digits Dataset
# ===============================

# 1️⃣ IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

# ===============================
# 2️⃣ DATA PREPARATION
# ===============================

# Load dataset
digits = load_digits()

X = digits.data      # Features
y = digits.target    # Labels

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data preparation completed.\n")

# ===============================
# 3️⃣ SUPERVISED LEARNING (KNN)
# ===============================

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

print("KNN Model Training Completed.\n")

# ===============================
# 4️⃣ EVALUATION METRICS
# ===============================

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===============================
# 5️⃣ UNSUPERVISED LEARNING (K-MEANS)
# ===============================

kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X)

clusters = kmeans.labels_

print("\nK-Means Clustering Completed.")

# ===============================
# 6️⃣ CLUSTER VISUALIZATION (PCA)
# ===============================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', s=15)
plt.title("K-Means Clustering (PCA Visualization)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar()
plt.show()

print("\nProject Completed Successfully.")
