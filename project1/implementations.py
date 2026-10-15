import numpy as np
from project1.helpers import sigmoid

def compute_mse(y, tx, w):
    """Compute the mean squared error loss (with the 1/2 factor).

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the feature matrix.
        w: numpy array of shape (D,), the model parameters.

    Returns:
        A scalar, the loss 1/(2N) * ||y - tx w||^2.
    """
    e = y - tx.dot(w)
    return 0.5 * np.mean(e**2)


def compute_mse_gradient(y, tx, w):
    """Compute the gradient of the MSE loss at w.
    
    Args:
        y: numpy array of shape (B,), the labels.
        tx: numpy array of shape (B, D), the feature matrix.
        w: numpy array of shape (D,), the model parameters.

    Returns:
        A numpy array of shape (D,), the gradient -1/B * tx^T (y - tx w).
    """
    e = y - tx.dot(w)
    return -tx.T.dot(e) / len(y)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the feature matrix.
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, the number of gradient steps.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the last weight vector.
        loss: scalar, the MSE loss at the last weight vector.
    """
    w = initial_w
    for _ in range(max_iters):
        # Full-batch gradient step
        w = w - gamma * compute_mse_gradient(y, tx, w)
    # Loss of the returned weights 
    loss = compute_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (mini-batch size 1).

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the feature matrix.
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, the number of stochastic gradient steps.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the last weight vector.
        loss: scalar, the MSE loss on the full dataset at the last weight vector.
    """
    w = initial_w
    n = len(y)
    for _ in range(max_iters):
        # Sample a single datapoint uniformly at random
        i = np.random.randint(n)
        # Slicing with i:i+1 keeps the 2D shape of tx for the gradient helper
        w = w - gamma * compute_mse_gradient(y[i : i + 1], tx[i : i + 1], w)
    # Loss of the returned weights, evaluated on the whole dataset
    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    
    # Solving normal equation using np.linalg.solve: (X.T @ X) @ w = X.T @ y
    A = tx.T @ tx
    b = tx.T @ y
    w_opti = np.linalg.solve(A,b) # allow to find x such that Ax = b
    mse = compute_mse(y, tx, w_opti)
    return w_opti, float(mse)


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """

    # Normal equation to solve for L2 regularization: (X.T@X + lambda*I)@w = X.T@y
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T @ tx + aI
    b = tx.T @ y
    return np.linalg.solve(a, b)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the feature matrix.
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, the number of gradient steps.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the final weights.
        loss: scalar, the MSE loss at the last weight vector.
    """

    # Initialize weights
    w = np.array(initial_w, dtype=float, copy=True)

    # Perform gradient descent
    for _ in range(max_iters):
        predictions = sigmoid(tx @ w)
        gradient = tx.T @ (predictions - y) / len(y)
        w = w - gamma * gradient

    # Compute the loss after the final update
    z = tx @ w
    loss = np.mean(np.logaddexp(0, z) - y * z)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic regression with L2 regularization.

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the feature matrix.
        lambda_: Regularization parameter.
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, the number of gradient steps.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the final weights.
        loss: scalar, the MSE loss at the last weight vector.
    """
    # Initialize weights
    w = np.array(initial_w, dtype=float, copy=True)

    # Perform gradient descent with L2 regularization
    for _ in range(max_iters):
        predictions = sigmoid(tx @ w)
        gradient = (
            tx.T @ (predictions - y) / len(y)
            + 2 * lambda_ * w
        )
        w = w - gamma * gradient

    # Compute the loss after the final update
    z = tx @ w
    # np.logaddexp(0, z) computes log(1 + exp(z)) in a numerically stable way
    loss = np.mean(np.logaddexp(0, z) - y * z)

    return w, loss