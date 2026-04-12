import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# ==============================
# A1: BASIC MODULES
# ==============================

def summation_unit(x, weights):
    return np.dot(x, weights[1:]) + weights[0]

# Activation Functions
def step(x):
    return 1 if x >= 0 else 0

def bipolar_step(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return max(0, x)

def leaky_relu(x):
    return x if x > 0 else 0.01 * x

def calculate_error(target, output):
    return target - output

# ==============================
# PERCEPTRON TRAINING
# ==============================

def train_perceptron(X, y, weights, lr, activation_func, max_epochs=1000):
    epochs = 0
    errors = []

    while epochs < max_epochs:
        total_error = 0

        for i in range(len(X)):
            net = summation_unit(X[i], weights)
            output = activation_func(net)

            error = calculate_error(y[i], output)
            total_error += error**2

            weights[1:] += lr * error * X[i]
            weights[0] += lr * error

        errors.append(total_error)

        if total_error <= 0.002:
            break

        epochs += 1

    return weights, errors, epochs

# ==============================
# DATASETS
# ==============================

def get_and_data():
    return np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,0,0,1])

def get_xor_data():
    return np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,1,1,0])

def get_customer_data():
    X = np.array([
        [20,6,2],[16,3,6],[27,6,2],[19,1,2],[24,4,2],
        [22,1,5],[15,4,2],[18,4,2],[21,1,4],[16,2,4]
    ])
    y = np.array([1,1,1,0,1,0,1,1,0,0])
    return X, y

# ==============================
# PLOT FUNCTION
# ==============================

def plot_error(errors, title):
    plt.figure()
    plt.plot(errors)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()

# ==============================
# A7: PSEUDO-INVERSE
# ==============================

def pseudo_inverse_solution(X, y):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    return np.linalg.pinv(X_bias).dot(y)

# ==============================
# A8: BACKPROPAGATION (MLP)
# ==============================

def sigmoid_derivative(x):
    return x * (1 - x)

def train_mlp(X, y, lr=0.05, epochs=1000):
    input_size = X.shape[1]
    hidden_size = 2
    output_size = 1

    W1 = np.random.randn(input_size, hidden_size)
    W2 = np.random.randn(hidden_size, output_size)

    for _ in range(epochs):
        hidden = sigmoid(np.dot(X, W1))
        output = sigmoid(np.dot(hidden, W2))

        error = y.reshape(-1,1) - output

        d_output = error * sigmoid_derivative(output)
        d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden)

        W2 += hidden.T.dot(d_output) * lr
        W1 += X.T.dot(d_hidden) * lr

    return W1, W2

# ==============================
# MAIN PROGRAM
# ==============================

if __name__ == "__main__":

    # A2: AND Gate
    X, y = get_and_data()
    weights = np.array([10, 0.2, -0.75], dtype=float)

    w, errors, epochs = train_perceptron(X, y, weights.copy(), 0.05, step)
    print("A2 AND -> Epochs:", epochs)
    plot_error(errors, "AND Gate - Step")

    # A3: Activation comparison
    activations = {
        "Bipolar": bipolar_step,
        "Sigmoid": sigmoid,
        "ReLU": relu
    }

    for name, func in activations.items():
        w, e, ep = train_perceptron(X, y, weights.copy(), 0.05, func)
        print(f"A3 {name} -> Epochs:", ep)

    # A4: Learning rate variation
    rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    epoch_list = []

    for lr in rates:
        _, _, ep = train_perceptron(X, y, weights.copy(), lr, step)
        epoch_list.append(ep)

    plt.figure()
    plt.plot(rates, epoch_list, marker='o')
    plt.title("Learning Rate vs Epochs")
    plt.xlabel("Learning Rate")
    plt.ylabel("Epochs")
    plt.show()

    # A5: XOR
    X_xor, y_xor = get_xor_data()
    _, _, ep = train_perceptron(X_xor, y_xor, weights.copy(), 0.05, step)
    print("A5 XOR Epochs:", ep)

    # A6: Customer Data
    X_c, y_c = get_customer_data()
    w, _, ep = train_perceptron(X_c, y_c, np.random.randn(4), 0.05, sigmoid)
    print("A6 Customer Epochs:", ep)

    # A7: Pseudo-inverse
    w_pinv = pseudo_inverse_solution(X_c, y_c)
    print("A7 Pseudo-inverse Weights:", w_pinv)

    # A8: Backpropagation
    W1, W2 = train_mlp(X, y)
    print("A8 MLP Weights:", W1, W2)

    # A11: sklearn MLP
    model = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', max_iter=1000)
    model.fit(X, y)
    print("A11 MLP Predictions:", model.predict(X))