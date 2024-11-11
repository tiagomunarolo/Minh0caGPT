import numpy as np
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from statsmodels.sandbox.distributions.mv_normal import np_log


class CustomLogisticRegression:

    def __init__(self, epochs: int, learning_rate=1e-3):
        self.epochs = epochs
        self.weights = None
        self.learning_rate = learning_rate
        self.Y_HAT = None
        self.X = None
        self.Y = None

    def predict(self, X, resolve=False):
        if not (X[:, 0] == 1).all():
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))
        Theta = X.dot(self.weights)
        Z = 1 / (1 + np.exp(-Theta))
        if resolve:
            return np.float64(Z > 0.5)
        return Z

    def compute_error(self):
        """
        Problem: min(y -yhat)
        Error function => -np.log(y_hat)*y + (1-y)*np.log(1-y_hat)
        """
        self.Y_HAT = self.predict(self.X)
        return -np.sum(self.Y * np.log(self.Y_HAT) + (1 - self.Y) * np_log(1 - self.Y_HAT))

    def gradient_descent(self) -> None:
        """
        E = -(log(y_hat)*y + (1-y)*log(1-y_hat))
        E = -(log(1/(1+exp(-W*X + B)))*y + (1-y)*np.log(1-(1+exp(-W*X + B)))

        E = A + B
        A = -log(1/(1+exp(-W*X + B)))*y
        B = -(1-y)*np.log(1-(1+exp(-W*X + B))
        dE/dW = (dA/dE)*dA/dW + (dB/dE)*dW/dB
        dA/dE = B
        dB/dE = A

        U =  1/(1+exp(-W*X + B)
        A = -log(U)*Y
        dA/dW = dA/dU*dU/dW
        dA/dU = -Y*1/U = -Y * (1/ (1/(1+exp(-W*X + B))
        ... dA/dW = -Y * (1/ (1/(1+exp(-W*X + B)) * -(1+exp(-W*X + B))**(-2) * - X * exp(-W*X + B)
        ==
        Z = exp(-W*X + B)
         -Y * (1/ (1/(1+Z)) * -(1+Z)**(-2) * - X * Z
         = -Y*X*Z/(1+Z)

        K = 1+exp(-W*X + B)
        U = K^-1
        dU/dW = dU/dK * dK/dW
        dU/dK = -1*(K)^-2 = -Kˆ(-2) = -(1+exp(-W*X + B))**(-2)
        ... dU/dW = -(1+exp(-W*X + B))**(-2) * - X * exp(-W*X + B)

        J= -W*X + B
        K = 1 + exp(J)
        dK/dW = dK/dJ * dJ/dW
        dK/dJ = exp(J)
        dJ/dW = -X

        dK/dW = - X * exp(-W*X + B)
        ...
        """
        diff = self.Y - self.Y_HAT
        ddw = self.X.T.dot(diff)
        self.weights += self.learning_rate * ddw

    def fit(self, X, Y, show=False):
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
        self.X = X
        self.Y = Y
        self.weights = np.zeros(X.shape[1]).reshape(-1, 1)
        current_error = np.inf
        for _ in range(self.epochs):
            error = self.compute_error()
            if error >= current_error:
                break
            current_error = error
            self.gradient_descent()
            if show and _ % 500 == 0:
                print(f"LOSS {_}: ", round(error, 3))


def sigmoid(X: np.float32, W: np.float32, B: np.float32):
    return 1 / (np.exp(-W * X + B) + 1)


def plot_sigmoid():
    # Define the range of x values from -10 to 10
    x = np.arange(-10, 11, 1)

    # Define a function y = x^2
    W = 1 / 2
    B = -5
    y = sigmoid(x, W, B)

    # Create the plot
    plt.plot(x, y, label="y = sigmoid(x)", color='blue', marker='o')

    # Add titles and labels
    plt.title('Plot of y = sigmoid(x)')
    plt.xlabel('x values')
    plt.ylabel('y values')

    # Add grid and legend
    plt.grid(True)
    plt.legend()

    # Display the plot
    plt.show()


def main():
    dataset = load_breast_cancer()
    # 30 features
    # 2 classes, binary classification
    X = dataset.data  # X (569. 30)
    X /= np.max(X, axis=0)
    Y = dataset.target.reshape(-1, 1)  # (569, 1)
    start = time()
    sklearn_lr = LogisticRegression()
    sklearn_lr.fit(X, Y)
    end = time()
    sk_time = round(end - start, 3)
    start = time()
    custom = CustomLogisticRegression(epochs=1000)
    custom.fit(X, Y)
    end = time()
    custom_time = round(end - start, 3)
    # print(lr.weights)
    # print("SKLEARN: ", w0, intercept)
    Y_hat = custom.predict(X, resolve=True)
    print("SKLEARN_TIME", sk_time)
    print("CUSTOM_TIME", custom_time)
    print("ACCURACY [SKLEARN]: ", round(accuracy_score(Y, sklearn_lr.predict(X)), 2))
    print("ACCURACY [MODEL]: ", round(accuracy_score(Y, Y_hat), 2))


if __name__ == '__main__':
    # plot_sigmoid()
    main()
