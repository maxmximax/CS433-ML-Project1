import numpy as np

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
