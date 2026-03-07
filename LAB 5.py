# ---------------------------------------------
# Load required libraries
# ---------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score


# ---------------------------------------------
# Load the dataset
# ---------------------------------------------
# The dataset contains GloVe word embeddings (glove_0 to glove_299)
# and a label column which we use as the target variable

df = pd.read_excel("Clarity_Text_student_teacher_with_glove.xlsx")

# Select only the embedding features
X = df[[col for col in df.columns if "glove_" in col]]

# Target column
y = df["Label"]


# ---------------------------------------------
# Split dataset into training and testing sets
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------------------------------------
# A1: Linear Regression using one attribute
# ---------------------------------------------
# We use only one feature (glove_0) to train the model

X_train_single = X_train[["glove_0"]]
X_test_single = X_test[["glove_0"]]

reg_model = LinearRegression()

reg_model.fit(X_train_single, y_train)

# Predictions
train_pred = reg_model.predict(X_train_single)
test_pred = reg_model.predict(X_test_single)


# ---------------------------------------------
# A2: Calculate evaluation metrics
# ---------------------------------------------
# MSE, RMSE, MAPE and R²

mse_train = mean_squared_error(y_train, train_pred)
rmse_train = np.sqrt(mse_train)
mape_train = np.mean(np.abs((y_train - train_pred) / (y_train + 1e-10))) * 100
r2_train = r2_score(y_train, train_pred)

mse_test = mean_squared_error(y_test, test_pred)
rmse_test = np.sqrt(mse_test)
mape_test = np.mean(np.abs((y_test - test_pred) / (y_test + 1e-10))) * 100
r2_test = r2_score(y_test, test_pred)

print("Training Metrics:")
print("MSE:", mse_train, "RMSE:", rmse_train, "MAPE:", mape_train, "R2:", r2_train)

print("\nTesting Metrics:")
print("MSE:", mse_test, "RMSE:", rmse_test, "MAPE:", mape_test, "R2:", r2_test)


# ---------------------------------------------
# A3: Linear Regression using all attributes
# ---------------------------------------------
# Now we train the model using all GloVe features

reg_all = LinearRegression()

reg_all.fit(X_train, y_train)

train_pred_all = reg_all.predict(X_train)
test_pred_all = reg_all.predict(X_test)

print("\nRegression using all features")
print("Train R2 Score:", r2_score(y_train, train_pred_all))
print("Test R2 Score:", r2_score(y_test, test_pred_all))


# ---------------------------------------------
# A4: Perform K-Means clustering
# ---------------------------------------------
# Clustering is performed only on the feature data

kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")

kmeans.fit(X_train)

cluster_labels = kmeans.labels_

print("\nCluster Centers:")
print(kmeans.cluster_centers_)


# ---------------------------------------------
# A5: Evaluate clustering performance
# ---------------------------------------------
sil_score = silhouette_score(X_train, cluster_labels)
ch_score = calinski_harabasz_score(X_train, cluster_labels)
db_score = davies_bouldin_score(X_train, cluster_labels)

print("\nClustering Evaluation Scores")
print("Silhouette Score:", sil_score)
print("Calinski-Harabasz Score:", ch_score)
print("Davies-Bouldin Index:", db_score)


# ---------------------------------------------
# A6: Evaluate clustering for different k values
# ---------------------------------------------
k_values = range(2, 10)

silhouette_scores = []

for k in k_values:

    kmeans = KMeans(n_clusters=k, random_state=42)

    labels = kmeans.fit_predict(X_train)

    score = silhouette_score(X_train, labels)

    silhouette_scores.append(score)

plt.plot(k_values, silhouette_scores)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs k")
plt.show()


# ---------------------------------------------
# A7: Elbow Method to find optimal clusters
# ---------------------------------------------
distortions = []

for k in range(2, 20):

    kmeans = KMeans(n_clusters=k, random_state=42)

    kmeans.fit(X_train)

    distortions.append(kmeans.inertia_)

plt.plot(range(2, 20), distortions)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Distortion")

plt.title("Elbow Method")

plt.show()