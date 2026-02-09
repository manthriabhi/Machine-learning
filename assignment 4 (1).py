#1
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


# Load the dataset
df = pd.read_excel("Clarity_Text_student_teacher_with_glove.xlsx")



# Separate input and output
# X = all GloVe features
X = df.filter(regex="glove_")

# y = clarity label
y = df["Label"]



# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)



# Create and train KNN model

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)


# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# Confusion Matrix
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)


# Performance Metrics (Train)
precision_train = precision_score(y_train, y_train_pred, average="weighted")
recall_train = recall_score(y_train, y_train_pred, average="weighted")
f1_train = f1_score(y_train, y_train_pred, average="weighted")


# Performance Metrics (Test)
precision_test = precision_score(y_test, y_test_pred, average="weighted")
recall_test = recall_score(y_test, y_test_pred, average="weighted")
f1_test = f1_score(y_test, y_test_pred, average="weighted")


# Print Resultsprint("Training Confusion Matrix:")
print(cm_train)

print("\nTesting Confusion Matrix:")
print(cm_test)

print("\nTraining Performance:")
print("Precision:", precision_train)
print("Recall   :", recall_train)
print("F1 Score :", f1_train)

print("\nTesting Performance:")
print("Precision:", precision_test)
print("Recall   :", recall_test)
print("F1 Score :", f1_test)

#2
import numpy as np


# ----------------------------------
# Mean Squared Error (MSE)
# ----------------------------------
def mse(y_true, y_pred):
    """Calculate Mean Squared Error"""

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    error = (y_true - y_pred) ** 2
    mse_value = np.mean(error)

    return mse_value


# ----------------------------------
# Root Mean Squared Error (RMSE)
# ----------------------------------
def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""

    mse_value = mse(y_true, y_pred)
    rmse_value = np.sqrt(mse_value)

    return rmse_value


# ----------------------------------
# Mean Absolute Percentage Error (MAPE)
# ----------------------------------
def mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    epsilon = 1e-10   # Avoid division by zero

    percentage_error = np.abs((y_true - y_pred) / (y_true + epsilon))
    mape_value = np.mean(percentage_error) * 100

    return mape_value


# ----------------------------------
# R-Squared Score (R2)
# ----------------------------------
def r2(y_true, y_pred):
    """Calculate R2 Score"""

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    rss = np.sum((y_true - y_pred) ** 2)

    r2_value = 1 - (rss / tss)

    return r2_value


# ----------------------------------
# Sample Data
# ----------------------------------
actual = [1, 55, 18]
predicted = [0.99, 54.99, 18]


# ----------------------------------
# Display Results
# ----------------------------------
print("Actual Values   :", actual)
print("Predicted Values:", predicted)
print()

print("MSE  :", round(mse(actual, predicted), 4))
print("RMSE :", round(rmse(actual, predicted), 4))
print("MAPE :", round(mape(actual, predicted), 4), "%")
print("R2   :", round(r2(actual, predicted), 4))

#3
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------
# Generate Random Numbers
# ----------------------------------
rnd = np.random.default_rng(seed=12)

# Create random matrix (2 rows, 20 columns)
X = np.random.randint(1, 10, size=(2, 20))


# ----------------------------------
# Create Labels
# ----------------------------------
y = []

for i in range(20):
    if X[1][i] // X[0][i] <= 0:
        y.append(1)
    else:
        y.append(0)


# ----------------------------------
# Separate Data into Two Classes
# ----------------------------------
blue = [[], []]
red = [[], []]

for i in range(20):

    if y[i] == 0:
        blue[0].append(X[0][i])
        blue[1].append(X[1][i])

    else:
        red[0].append(X[0][i])
        red[1].append(X[1][i])


# ----------------------------------
# Plot the Data
# ----------------------------------
plt.figure(figsize=(7, 5))

plt.scatter(blue[0], blue[1], color="blue", label="Class 0")
plt.scatter(red[0], red[1], color="red", label="Class 1")

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Scatter Plot of Two Classes")

plt.legend()
plt.grid(True)

plt.show()


#4
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier


# ----------------------------------
# Generate Training Data
# ----------------------------------
np.random.seed(12)

# Create random data (2 rows, 20 columns)
X = np.random.randint(1, 10, size=(2, 20))


# ----------------------------------
# Create Labels
# ----------------------------------
y = []

for i in range(20):
    if X[1][i] > X[0][i]:
        y.append(1)   # Class 1
    else:
        y.append(0)   # Class 0


# ----------------------------------
# Plot Training Data
# ----------------------------------
blue = [[], []]
red = [[], []]

for i in range(20):

    if y[i] == 0:
        blue[0].append(X[0][i])
        blue[1].append(X[1][i])

    else:
        red[0].append(X[0][i])
        red[1].append(X[1][i])


plt.figure(figsize=(8, 6))

plt.scatter(blue[0], blue[1], color="blue", label="Class 0 (Train)")
plt.scatter(red[0], red[1], color="red", label="Class 1 (Train)")


# ----------------------------------
# Generate Test Data
# ----------------------------------
xtest = np.random.uniform(
    low=1.0,
    high=10.0,
    size=(10000, 2)
)


# ----------------------------------
# Train KNN Classifier
# ----------------------------------
classifier = KNeighborsClassifier(n_neighbors=3)

# Transpose X so shape becomes (samples, features)
classifier.fit(X.T, y)


# ----------------------------------
# Predict Test Data
# ----------------------------------
ytest = classifier.predict(xtest)


# ----------------------------------
# Plot Test Predictions
# ----------------------------------
plt.scatter(
    xtest[ytest == 0, 0],
    xtest[ytest == 0, 1],
    color="lightblue",
    s=10,
    label="Class 0 (Test)"
)

plt.scatter(
    xtest[ytest == 1, 0],
    xtest[ytest == 1, 1],
    color="pink",
    s=10,
    label="Class 1 (Test)"
)


# ----------------------------------
# Graph Settings
# ----------------------------------
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("KNN Classification with Test Data")

plt.legend()
plt.grid(True)

plt.show()

#5
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier


# ----------------------------------
# Generate Training Data
# ----------------------------------
np.random.seed(12)

X = np.random.randint(1, 10, size=(2, 20))


# ----------------------------------
# Create Labels
# ----------------------------------
y = []

for i in range(20):
    if X[1][i] > X[0][i]:
        y.append(1)
    else:
        y.append(0)


# ----------------------------------
# Function to Train and Plot KNN
# ----------------------------------
def plot_knn(k_value, position):

    # Generate Test Data
    xtest = np.random.uniform(1.0, 10.0, size=(10000, 2))

    # Train KNN
    classifier = KNeighborsClassifier(n_neighbors=k_value)
    classifier.fit(X.T, y)

    # Predict
    ytest = classifier.predict(xtest)

    # Plot
    plt.subplot(1, 3, position)

    plt.scatter(
        xtest[ytest == 0, 0],
        xtest[ytest == 0, 1],
        color="blue",
        s=8,
        label="Class 0"
    )

    plt.scatter(
        xtest[ytest == 1, 0],
        xtest[ytest == 1, 1],
        color="red",
        s=8,
        label="Class 1"
    )

    # Plot training points
    for i in range(len(y)):
        if y[i] == 0:
            plt.scatter(X[0][i], X[1][i], color="black", s=50)
        else:
            plt.scatter(X[0][i], X[1][i], color="green", s=50)

    plt.title(f"KNN (K = {k_value})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)


# ----------------------------------
# Plot for K = 2, 4, 5
# ----------------------------------
plt.figure(figsize=(15, 5))

plot_knn(2, 1)
plot_knn(4, 2)
plot_knn(5, 3)

plt.suptitle("KNN Classification for Different K Values")
plt.show()

#6
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier


# Read the dataset
df = pd.read_excel("Clarity_Text_student_teacher_with_glove.xlsx")


# ----------------------------------
# Convert label into binary form
# 0 -> Low / Medium clarity
# 1 -> High clarity
# ----------------------------------
df["BinaryLabel"] = np.where(df["Label"] <= 2, 0, 1)


# ----------------------------------
# Select only first two features
# (used only for visualization)
# ----------------------------------
X = df[["glove_0", "glove_1"]]
y = df["BinaryLabel"]



# Create and train KNN model

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)



# Create grid for decision boundary
x_min = X.iloc[:, 0].min() - 0.1
x_max = X.iloc[:, 0].max() + 0.1

y_min = X.iloc[:, 1].min() - 0.1
y_max = X.iloc[:, 1].max() + 0.1


xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.01),
    np.arange(y_min, y_max, 0.01)
)


# Predict values for grid points
grid_points = np.c_[xx.ravel(), yy.ravel()]

Z = model.predict(grid_points)
Z = Z.reshape(xx.shape)


# Plot the result
plt.figure(figsize=(8, 6))

# Background prediction area
plt.scatter(xx, yy, c=Z, alpha=0.2)

# Actual data points
plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=y,
    edgecolors="k"
)

plt.xlabel("glove_0")
plt.ylabel("glove_1")
plt.title("KNN Decision Boundary for Speech Clarity")

plt.show()

#7
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Read Excel file
df = pd.read_excel("Clarity_Text_student_teacher_with_glove.xlsx")

# Input features (all glove columns)
X = df.filter(regex="glove_")

# Output label
y = df["Label"]

# Range of K values (1 to 20)
param_grid = {"n_neighbors": np.arange(1, 21)}

# Create KNN model
knn = KNeighborsClassifier()

# Apply GridSearch
grid = GridSearchCV(
    knn,
    param_grid,
    cv=5,
    scoring="f1_weighted"
)

# Train model
grid.fit(X, y)

# Print best result
print("Best K value:", grid.best_params_)
print("Best Score:", grid.best_score_)
    
