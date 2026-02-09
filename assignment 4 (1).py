import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report


# ---------------------------------
# Load Speaker Clarity Dataset
# ---------------------------------
def load_audio_dataset(file_path):

    df = pd.read_excel(file_path)

    # Select TWO features for visualization
    X = df.iloc[:, 0:2]        # First two feature columns

    # Target (Clarity Label: 0 = unclear, 1 = clear)
    y = df["clarity_label"]   # Change if column name is different

    return X, y


# ---------------------------------
# A1: Classification Metrics
# ---------------------------------
def evaluate_model(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    return cm, report


# ---------------------------------
# Plot Training Data (A3, A6)
# ---------------------------------
def plot_training_data(X, y, title):

    plt.figure(figsize=(8,6))

    plt.scatter(
        X[y==0].iloc[:,0],
        X[y==0].iloc[:,1],
        c="blue",
        label="Unclear"
    )

    plt.scatter(
        X[y==1].iloc[:,0],
        X[y==1].iloc[:,1],
        c="red",
        label="Clear"
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)

    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------------
# Generate Test Grid (A4)
# ---------------------------------
def generate_test_grid(X):

    x_min, x_max = X.iloc[:,0].min()-1, X.iloc[:,0].max()+1
    y_min, y_max = X.iloc[:,1].min()-1, X.iloc[:,1].max()+1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    return grid, xx, yy


# ---------------------------------
# Train kNN (A4)
# ---------------------------------
def train_knn_classifier(X_train, y_train, k=3):

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    return knn


# ---------------------------------
# Plot Decision Boundary
# ---------------------------------
def plot_knn_results(model, X, y, title):

    grid, xx, yy = generate_test_grid(X)

    preds = model.predict(grid)
    preds = preds.reshape(xx.shape)

    plt.figure(figsize=(8,6))

    plt.contourf(xx, yy, preds, alpha=0.3)

    plt.scatter(
        X.iloc[:,0],
        X.iloc[:,1],
        c=y,
        edgecolors="k"
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)

    plt.show()


# ---------------------------------
# A5: Compare k Values
# ---------------------------------
def compare_k_values(X_train, y_train, k_list):

    for k in k_list:

        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        plot_knn_results(
            model,
            X_train,
            y_train,
            f"kNN Boundary (k={k})"
        )


# ---------------------------------
# A7: Hyperparameter Tuning
# ---------------------------------
def tune_k_value(X_train, y_train):

    params = {
        "n_neighbors": list(range(1, 21))
    }

    knn = KNeighborsClassifier()

    grid = GridSearchCV(
        knn,
        params,
        cv=5,
        scoring="accuracy"
    )

    grid.fit(X_train, y_train)

    return grid.best_params_, grid.best_score_


# ---------------------------------
# MAIN PROGRAM
# ---------------------------------
def main():

    # Load Dataset
    file_path = "Clarity_Text_student_teacher_with_glove.xlsx"

    X, y = load_audio_dataset(file_path)


    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )


    # ---------------- A3 / A6 ----------------
    print("\nA3 / A6: Plotting Training Data")

    plot_training_data(
        X_train,
        y_train,
        "Speaker Clarity Training Data"
    )


    # ---------------- A4 ----------------
    print("\nA4: Training kNN (k=3)")

    knn = train_knn_classifier(X_train, y_train, k=3)

    plot_knn_results(
        knn,
        X_train,
        y_train,
        "kNN Classification (k=3)"
    )


    # ---------------- A1 ----------------
    print("\nA1: Performance Evaluation")

    train_pred = knn.predict(X_train)
    test_pred = knn.predict(X_test)


    cm_train, report_train = evaluate_model(y_train, train_pred)
    cm_test, report_test = evaluate_model(y_test, test_pred)


    print("\nTraining Confusion Matrix")
    print(cm_train)

    print("\nTraining Classification Report")
    print(report_train)


    print("\nTest Confusion Matrix")
    print(cm_test)

    print("\nTest Classification Report")
    print(report_test)


    #  A5 ----------------
    print("\nA5: Comparing k Values")

    k_values = [1, 3, 5, 7, 10, 15]

    compare_k_values(X_train, y_train, k_values)


  # A7 
    print("\nA7: Finding Best k")

    best_k, best_score = tune_k_value(X_train, y_train)

    print("Best k =", best_k["n_neighbors"])
    print("Best Accuracy =", best_score)


   
    final_knn = KNeighborsClassifier(
        n_neighbors=best_k["n_neighbors"]
    )

    final_knn.fit(X_train, y_train)

    plot_knn_results(
        final_knn,
        X_train,
        y_train,
        "Best kNN Model"
    )


if __name__ == "__main__":
    main()

