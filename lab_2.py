
# IMPORTS

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder



# 1

def load_purchase_data(file_path):
    df = pd.read_excel(file_path, sheet_name="Purchase data")
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = df["Payment (Rs)"].values.reshape(-1, 1)
    return X, y


def calculate_rank(matrix):
    return np.linalg.matrix_rank(matrix)


def calculate_cost_pinv(X, y):
    X_pinv = np.linalg.pinv(X)
    return X_pinv @ y
# 2
def create_labels(y):
    return np.where(y > 200, 1, 0)

def train_classifier(X, labels):
    model = LogisticRegression()
    model.fit(X, labels)
    return model

# 3

def load_stock_data(file_path):
    return pd.read_excel(file_path, sheet_name="IRCTC Stock Price")


def manual_mean(data):
    total = 0
    for value in data:
        total += value
    return total / len(data)


def manual_variance(data):
    mean = manual_mean(data)
    total = 0
    for value in data:
        total += (value - mean) ** 2
    return total / len(data)


def execution_time(func, data):
    times = []
    for _ in range(10):
        start = time.time()
        func(data)
        times.append(time.time() - start)
    return sum(times) / len(times)



#4

def load_thyroid_data(file_path):
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")


def explore_data(df):
    info = df.info()
    description = df.describe(include="all")
    missing = df.isnull().sum()
    return info, description, missing

# 5

def jaccard_smc(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))

    jc = f11 / (f11 + f10 + f01)
    smc = (f11 + f00) / (f11 + f00 + f10 + f01)
    return jc, smc



#6
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))



# 7

def similarity_matrix(data):
    n = len(data)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i][j] = cosine_similarity(data[i], data[j])
    return sim



# 8
def impute_data(df):
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df


# 9
def normalize_data(df):
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df



# MAIN FUNCTION

def main():
    file_path = "Lab Session Data.xlsx"

    # 1
    X, y = load_purchase_data(file_path)
    print("Dimensionality:", X.shape[1])
    print("Number of vectors:", X.shape[0])
    print("Rank of matrix:", calculate_rank(X))
    print("Cost of products:\n", calculate_cost_pinv(X, y))

    # 2
    labels = create_labels(y)
    model = train_classifier(X, labels)
    print("Classifier trained successfully")

    # 3
    stock_df = load_stock_data(file_path)
    price = stock_df.iloc[:, 3]

    print("Mean (numpy):", np.mean(price))
    print("Variance (numpy):", np.var(price))
    print("Mean (manual):", manual_mean(price))
    print("Variance (manual):", manual_variance(price))
    print("Time numpy mean:", execution_time(np.mean, price))
    print("Time manual mean:", execution_time(manual_mean, price))

    wed_prices = stock_df[stock_df["Day"] == "Wednesday"].iloc[:, 3]
    print("Wednesday Mean:", np.mean(wed_prices))

    april_prices = stock_df[stock_df["Month"] == "Apr"].iloc[:, 3]
    print("April Mean:", np.mean(april_prices))

    loss_prob = len(list(filter(lambda x: x < 0, stock_df["Chg%"]))) / len(stock_df)
    print("Probability of loss:", loss_prob)

    # Scatter plot
    plt.scatter(stock_df["Day"], stock_df["Chg%"])
    plt.title("Chg% vs Day")
    plt.show()

    #4
    thyroid_df = load_thyroid_data(file_path)
    _, _, missing = explore_data(thyroid_df)
    print("Missing values:\n", missing)

    #5
    v1 = np.array([1, 0, 1, 1])
    v2 = np.array([1, 1, 0, 1])
    jc, smc = jaccard_smc(v1, v2)
    print("Jaccard:", jc, "SMC:", smc)

    # 6
    print("Cosine similarity:", cosine_similarity(v1, v2))

    # 7
    data = np.random.randint(0, 2, (20, 5))
    sim = similarity_matrix(data)
    sns.heatmap(sim, annot=True)
    plt.title("Similarity Heatmap")
    plt.show()



# RUN
if __name__ == "__main__":
    main()

