# ===============================
# LAB 06 - DECISION TREE FROM SCRATCH
# ===============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ===============================
# LOAD DATASET
# ===============================
file_path = "Clarity_Text_student_teacher_with_glove.xlsx"
df = pd.read_excel(file_path)

print("Dataset shape:", df.shape)
print(df.head())

# ===============================
# SELECT FEATURES & TARGET
# ===============================
# Assuming:
# Features = glove_0 to glove_299
# Target = Label

X = df.loc[:, "glove_0":"glove_299"]
y = df["Label"]

# ===============================
# A1: ENTROPY FUNCTION
# ===============================
def entropy(y):
    counts = np.bincount(y)
    probs = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

print("Entropy:", entropy(y))

# ===============================
# EQUAL WIDTH BINNING FUNCTION
# ===============================
def equal_width_binning(data, bins=4):
    return pd.cut(data, bins=bins, labels=False)

# ===============================
# A2: GINI INDEX FUNCTION
# ===============================
def gini(y):
    counts = np.bincount(y)
    probs = counts / len(y)
    return 1 - np.sum(probs**2)

print("Gini Index:", gini(y))

# ===============================
# A3: INFORMATION GAIN
# ===============================
def information_gain(X_col, y):
    total_entropy = entropy(y)
    values = np.unique(X_col)

    weighted_entropy = 0
    for v in values:
        subset_y = y[X_col == v]
        weighted_entropy += (len(subset_y) / len(y)) * entropy(subset_y)

    return total_entropy - weighted_entropy

# ===============================
# ROOT NODE DETECTION
# ===============================
def best_feature(X, y):
    gains = {}

    for col in X.columns:
        binned = equal_width_binning(X[col])
        gains[col] = information_gain(binned.values, y.values)

    best = max(gains, key=gains.get)
    return best, gains

best_feat, gains = best_feature(X, y)
print("Best Feature (Root Node):", best_feat)

# ===============================
# A5: SIMPLE DECISION TREE (CUSTOM)
# ===============================
class SimpleDecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        if len(set(y)) == 1:
            return y.iloc[0]

        if depth >= self.max_depth:
            return Counter(y).most_common(1)[0][0]

        best_feat, _ = best_feature(X, y)

        tree = {best_feat: {}}

        binned = equal_width_binning(X[best_feat])

        for val in np.unique(binned):
            subset_X = X[binned == val]
            subset_y = y[binned == val]

            if len(subset_y) == 0:
                tree[best_feat][val] = Counter(y).most_common(1)[0][0]
            else:
                tree[best_feat][val] = self.fit(subset_X, subset_y, depth + 1)

        return tree

# Train custom tree
tree_model = SimpleDecisionTree(max_depth=3)
tree = tree_model.fit(X, y)

print("Custom Decision Tree:")
print(tree)

# ===============================
# A6: VISUALIZATION USING SKLEARN
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

plt.figure(figsize=(15, 8))
plot_tree(clf, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# ===============================
# A7: DECISION BOUNDARY (2 FEATURES)
# ===============================
# Take only 2 features
X2 = X[["glove_0", "glove_1"]]

# Train model
clf2 = DecisionTreeClassifier(max_depth=3)
clf2.fit(X2, y)

# Create mesh
x_min, x_max = X2.iloc[:, 0].min() - 1, X2.iloc[:, 0].max() + 1
y_min, y_max = X2.iloc[:, 1].min() - 1, X2.iloc[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X2.iloc[:, 0], X2.iloc[:, 1], c=y, edgecolor='k')
plt.xlabel("glove_0")
plt.ylabel("glove_1")
plt.title("Decision Boundary (2 Features)")
plt.show()