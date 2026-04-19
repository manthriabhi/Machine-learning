# ===============================
# REMOVE ALL WARNINGS (IMPORTANT)
# ===============================
import warnings
warnings.simplefilter("ignore")

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from lime.lime_tabular import LimeTabularExplainer


# ===============================
# LOAD DATA
# ===============================
def load_dataset():
    file_path = os.path.join(os.path.dirname(__file__), "dataset.xlsx")

    if not os.path.exists(file_path):
        print("ERROR: dataset.xlsx not found")
        return None

    return pd.read_excel(file_path)


# ===============================
# PREPROCESS
# ===============================
def preprocess(data):
    features = [col for col in data.columns if col.startswith("glove_")]

    X = data[features]
    y = data["Label"]

    X = X.apply(pd.to_numeric, errors="coerce")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return X, y, encoder


# ===============================
# BUILD MODEL
# ===============================
def build_pipeline():

    base_models = [
        ("rf", RandomForestClassifier(n_estimators=20)),
        ("et", ExtraTreesClassifier(n_estimators=20)),
        ("dt", DecisionTreeClassifier(max_depth=10))
    ]

    final_model = RandomForestClassifier(n_estimators=20)

    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=final_model,
        cv=3
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer()),
        ("variance", VarianceThreshold()),
        ("scaler", StandardScaler()),
        ("model", stack)
    ])

    return pipeline


# ===============================
# TRAIN + EVALUATE
# ===============================
def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test, encoder):

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\n===== RESULTS =====")
    print("Accuracy:", round(acc, 4))

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, preds))


# ===============================
# LIME
# ===============================
def run_lime(pipeline, X_train, X_test, encoder):

    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=[str(i) for i in encoder.classes_],
        mode="classification"
    )

    exp = explainer.explain_instance(
        X_test.iloc[0].values,
        pipeline.predict_proba
    )

    exp.save_to_file("lime_explanation.html")


# ===============================
# MAIN
# ===============================
def main():

    data = load_dataset()
    if data is None:
        return

    X, y, encoder = preprocess(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()

    train_and_evaluate(pipeline, X_train, X_test, y_train, y_test, encoder)

    run_lime(pipeline, X_train, X_test, encoder)

    print("\nLIME saved → lime_explanation.html")


# ===============================
if __name__ == "__main__":
    main()