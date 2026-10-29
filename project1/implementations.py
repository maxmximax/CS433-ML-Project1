import numpy as np



def sigmoid(z):
    """Compute the sigmoid without exponential overflow.

    Args:
        z: Input array.

    Returns:
        Sigmoid of the input array.
    """
    # exp(-|z|) never overflows; pick the matching formula depending on the sign of z
    exp_neg_abs = np.exp(-np.abs(z))
    return np.where(
        z >= 0,
        1.0 / (1.0 + exp_neg_abs),
        exp_neg_abs / (1.0 + exp_neg_abs),
    )


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


#---------Helper funcction start for logistic regression-----------

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    N = y.shape[0]
    pred = sigmoid(tx @ w)
    loss = -1/N * (y.T @ np.log(pred) + (1 - y).T @ np.log(1 - pred))
    return loss 

def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

      Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)

    Returns:
        a vector of shape (D,)

    >>> np.set_printoptions(8)
    >>> y = np.array([0., 1.])
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([0.1, 0.2, 0.3])
    >>> calculate_gradient(y, tx, w)
    array([-0.10370763,  0.2067104 ,  0.51712843])
    """

    N = y.shape[0]
    pred = sigmoid(tx @ w)
    grad = 1/N * (tx.T  @ (pred - y))
    return grad


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    loss = calculate_loss(y,tx,w)
    loss_grad = calculate_gradient(y,tx,w)
    w_new = w - gamma*loss_grad

    return loss, w_new

#_-----------------------Helper functions end for logistic regression----------------

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
        w: optimal weights, numpy array of shape (D,), D is the number of features.
        loss: scalar, the MSE loss.
    """
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.pinv(A) @ b
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar, the MSE loss.
    """

    # Normal equation to solve for L2 regularization: (X.T@X + lambda*I)@w = X.T@y
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T @ tx + aI
    b = tx.T @ y
    w_opti = np.linalg.solve(a, b)
    loss_mse = compute_mse(y, tx, w_opti)
    return w_opti, loss_mse


def logistic_regression_gauthier(y, tx, initial_w, max_iters, gamma):
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

    # init parameters
    threshold = 1e-8
    losses = []

    # build tx
    w = np.array(initial_w, dtype=float, copy=True)  # shape (D,)

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    print("loss={l}".format(l=calculate_loss(y, tx, w)))
    
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularized logistic regression using gradient descent.

    Args:
        y: labels in {0, 1}, shape (N,).
        tx: feature matrix, shape (N, D).
        lambda_: regularization strength.
        initial_w: initial weights, shape (D,).
        max_iters: number of gradient steps.
        gamma: learning rate.

    Returns:
        w: final weights, shape (D,).
        loss: mean logistic loss, excluding the penalty.
    """
    w = np.array(initial_w, dtype=float, copy=True)

    for iteration in range(max_iters):
        z = tx @ w

        # Regularize weights
        gradient = tx.T @ (sigmoid(z) - y) / len(y)
        gradient += 2 * lambda_ * w

        w -= gamma * gradient

    # Return the unpenalized loss at the final weights.
    z = tx @ w
    loss = np.mean(np.logaddexp(0, z) - y * z)

    return w, float(loss)