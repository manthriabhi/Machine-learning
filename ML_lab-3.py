import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import minkowski


# A1

def dot_product(vec1, vec2):
    """Calculate dot product of two vectors manually"""
    total = 0
    for i in range(len(vec1)):
        total += vec1[i] * vec2[i]
    return total


def euclidean_norm(vector):
    """Compute Euclidean norm (magnitude) of a vector"""
    squared_sum = 0
    for value in vector:
        squared_sum += value ** 2
    return squared_sum ** 0.5


# A2

def mean(values):
    """Calculate mean of a list"""
    return sum(values) / len(values)


def variance(values):
    """Calculate variance of a list"""
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)


def standard_deviation(values):
    """Calculate standard deviation"""
    return variance(values) ** 0.5


def dataset_statistics(data):
    """Return mean and std deviation for each feature"""
    return np.mean(data, axis=0), np.std(data, axis=0)


def centroid(class_samples):
    """Compute centroid of a class"""
    return np.mean(class_samples, axis=0)



# A3

def feature_statistics(feature):
    """Return mean and variance of a feature"""
    return mean(feature), variance(feature)


# A4

def minkowski_distance(vec1, vec2, p):
    """Custom Minkowski distance implementation"""
    total = 0
    for i in range(len(vec1)):
        total += abs(vec1[i] - vec2[i]) ** p
    return total ** (1 / p)


def euclidean_distance(vec1, vec2):
    """Compute Euclidean distance using NumPy"""
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


# A10:

def knn_predict(X_train, y_train, test_sample, k):
    """Predict class using custom KNN"""
    distance_label_pairs = []

    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], test_sample)
        distance_label_pairs.append((dist, y_train[i]))

    # Sort by distance
    distance_label_pairs.sort(key=lambda x: x[0])

    # Get k nearest labels
    nearest_labels = [label for _, label in distance_label_pairs[:k]]

    # Majority vote
    return max(set(nearest_labels), key=nearest_labels.count)


# A13

def classification_metrics(true_labels, predicted_labels):
    """Calculate confusion matrix and evaluation metrics"""
    cm = confusion_matrix(true_labels, predicted_labels)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    return cm, accuracy, precision, recall, f1_score


# MAIN PROGRAM

np.random.seed(0)

# Generate synthetic data
class_0 = np.random.normal(2, 1, (50, 2))
class_1 = np.random.normal(6, 1, (50, 2))

X = np.vstack((class_0, class_1))
y = np.array([0] * 50 + [1] * 50)

# Sample vectors
vec_a = X[0]
vec_b = X[1]

# Dot product comparison
dp_custom = dot_product(vec_a, vec_b)
dp_numpy = np.dot(vec_a, vec_b)

# Norm comparison
norm_custom = euclidean_norm(vec_a)
norm_numpy = np.linalg.norm(vec_a)

# Centroid distance
centroid_0 = centroid(class_0)
centroid_1 = centroid(class_1)
centroid_dist = np.linalg.norm(centroid_0 - centroid_1)

# Histogram of first feature
feature_1 = X[:, 0]
plt.hist(feature_1, bins=10)
plt.title("Histogram of Feature 1")
plt.show()

# Minkowski distance vs p
p_values = range(1, 11)
minkowski_distances = [minkowski_distance(vec_a, vec_b, p) for p in p_values]

plt.plot(p_values, minkowski_distances, marker='o')
plt.xlabel("p value")
plt.ylabel("Distance")
plt.title("Minkowski Distance vs p")
plt.show()

# Compare with SciPy
custom_minkowski = minkowski_distance(vec_a, vec_b, 3)
scipy_minkowski = minkowski(vec_a, vec_b, 3)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Sklearn KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

accuracy_sklearn = knn_model.score(X_test, y_test)
predictions_sklearn = knn_model.predict(X_test)

# Custom KNN predictions
predictions_custom = np.array([
    knn_predict(X_train, y_train, sample, 3) for sample in X_test
])

# Metrics
cm, acc, prec, rec, f1 = classification_metrics(y_test, predictions_sklearn)

# Accuracy vs K plot
k_values = range(1, 12)
accuracies = []

for k in k_values:
    preds = np.array([
        knn_predict(X_train, y_train, sample, k) for sample in X_test
    ])
    accuracies.append(np.mean(preds == y_test))

plt.plot(k_values, accuracies, marker='o')
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K in Custom KNN")
plt.show()
