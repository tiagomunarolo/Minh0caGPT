import numpy as np

np.random.seed(0)


class MultipleLinearRegression:
    """
    A class that implements a simple linear regression model using gradient descent.
    """

    def __init__(self, epochs: int, learning_rate: float = 1e-3):
        """
        Initialize the LinearRegression model.

        Parameters:
        epochs (int): Number of epochs to run gradient descent.
        learning_rate (float): The learning rate for gradient descent.
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.slopes = None
        self.intercept = None

    @staticmethod
    def compute_mse_error(y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Compute Mean Squared Error (MSE) between actual and predicted values.

        Parameters:
        y (np.ndarray): Actual values.
        y_hat (np.ndarray): Predicted values.

        Returns:
        float: The MSE.
        """
        return np.mean((y - y_hat) ** 2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output values for input X.

        Parameters:
        X (np.ndarray): Input features.

        Returns:
        np.ndarray: Predicted values.
        """
        return self.slopes.dot(X.T) + self.intercept

    def gradient_descent(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Perform one step of gradient descent to update the slope and intercept.

        Parameters:
        X (np.ndarray): Input features.
        Y (np.ndarray): Target values.
        """
        N = len(X)
        y_hat = self.predict(X)

        # Compute gradients
        slope_gradient = (-2 / N) * np.dot((Y - y_hat), X)
        intercept_gradient = (-2 / N) * np.sum(Y - y_hat)

        # Update parameters
        self.slopes -= self.learning_rate * slope_gradient
        self.intercept -= self.learning_rate * intercept_gradient

    def show_values(self, epoch: int, error: float):
        print(f"EPOCH [{epoch}]: MSE = {error:.3f}")
        slopes = [round(x, 2) for x in self.slopes.reshape(-1)]
        intercept = round(self.intercept[0][0], 2)
        print(f"Updated slope = {slopes}, intercept = {intercept}")
        predicted = self.predict(np.asarray([[1, 3, 10]]))[0][0]
        print(f"Predicted y(1, 3, 10) = {predicted:.2f}\n")

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Train the linear regression model using gradient descent.

        Parameters:
        X (np.ndarray): Input features.
        Y (np.ndarray): Target values.
        """
        current_error = np.inf
        self.slopes = np.zeros((1, X.shape[1]))
        self.intercept = np.zeros((1, 1))

        for epoch in range(self.epochs):
            y_hat = self.predict(X)
            error = self.compute_mse_error(Y, y_hat)

            if error >= current_error:
                break  # Stop if error increases

            current_error = error
            self.gradient_descent(X, Y)

            if epoch % 1000 == 0:
                self.show_values(epoch, error)
        self.show_values(self.epochs, current_error)


def main():
    # X
    # test = y = 2 * X1 - 3 * X2 + 9 *X3 + 12
    EPOCHS = 100000
    B = np.asarray([12]).reshape(1, 1)
    W = np.asarray([2, -3, 9]).reshape(1, 3)
    X = np.asarray([
        [1, 3, 10],
        [2, 4, 5],
        [3, -1, 8],
        [4, 6, 9],
        [5, 0, 1],
        [6, 8, 9],
        [7, 1, 1]]
    )
    Y = W.dot(X.T) + B
    lr = MultipleLinearRegression(epochs=EPOCHS)
    lr.fit(X, Y)


if __name__ == '__main__':
    main()
