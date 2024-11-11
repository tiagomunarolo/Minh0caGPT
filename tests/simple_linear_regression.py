import numpy as np

np.random.seed(0)


class LinearRegression:
    """
    A class that implements a simple linear regression model using gradient descent.
    """

    def __init__(self, epochs: int, slope: float = 0, intercept: float = 0, learning_rate: float = 1e-3):
        """
        Initialize the LinearRegression model.

        Parameters:
        epochs (int): Number of epochs to run gradient descent.
        slope (float): Initial slope (m) of the linear equation y = m*x + b.
        intercept (float): Initial intercept (b) of the linear equation y = m*x + b.
        learning_rate (float): The learning rate for gradient descent.
        """
        self.slope = slope
        self.intercept = intercept
        self.epochs = epochs
        self.learning_rate = learning_rate

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
        return self.slope * X + self.intercept

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
        slope_gradient = (-2 / N) * np.dot(X, (Y - y_hat))
        intercept_gradient = (-2 / N) * np.sum(Y - y_hat)

        # Update parameters
        self.slope -= self.learning_rate * slope_gradient
        self.intercept -= self.learning_rate * intercept_gradient

    def show_values(self, epoch: int, error: float):
        print(f"EPOCH [{epoch}]: MSE = {error:.3f}")
        print(f"Updated slope = {self.slope:.2f}, intercept = {self.intercept:.2f}")
        predicted = self.predict(np.asarray([4]))[0]
        print(f"Predicted y(4) = {predicted:.2f}\n")

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Train the linear regression model using gradient descent.

        Parameters:
        X (np.ndarray): Input features.
        Y (np.ndarray): Target values.
        """
        current_error = np.inf

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
    EPOCHS = 100000
    X = np.asarray([1, 2, 3, 4, 5, 6, 7])
    Y = 3 * X + 5

    lr = LinearRegression(epochs=EPOCHS)
    lr.fit(X, Y)


if __name__ == '__main__':
    main()
