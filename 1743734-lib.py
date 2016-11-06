import numpy as np


def innprod(x, y):
    """
    Inner product of matrix X and vector Y.
    :param x: Data matrix.
    :param y: Vector of "true" values.
    :return: Returns the inner product of every row of X with y.
    """
    if x.shape[-1] == y.shape[0]:
        return (x * y).sum(axis=-1)
    else:
        return np.nan


def grad(y, c, x):
    """
    Returns the gradient of f evaluated in x
    :param y: Vector of "true" values.
    :param c: Coefficients.
    :param x: Point to evaluate in the gradient.
    :return: Returns the gradient evaluated in x.
    """
    return -2.0*innprod(c.T, y - innprod(c, x))


def rss(y, x, w):
    """
    The residual sum of squares. Simply, is our objective function we are trying to minimize.
    :param y: Vector of " true " values.
    :param x: Data matrix.
    :param w: Weights of the linear model.
    :return: The residual sum of squares given w.
    """
    a = y - innprod(x, w)
    return innprod(a.T, a)


def descent(y, x, alpha=1e-3, itr=1e2, eps=1e-6):
    """
    Gradient descent algorithm. Method for minimize RSS given w.
    :param y: Vector of "true" valuies.
    :param x: Data matrix.
    :param alpha: Step for the gradient.
    :param itr: Max number of iterations allowed.
    :param eps: Epsilon for the relative error between solutions t+1 and t.
    :return: The vector w that minimizes RSS iff converges.
    """
    w = np.ones(x[0].size)
    while itr > 0:
        w1 = w - alpha * grad(y, x, w)
        if np.linalg.norm(w1 - w, 1)/np.linalg.norm(w, 1) < eps:
            return w1
        w = w1
        itr -= 1
    return w


def r2(y, c, x):
    """
    Coefficient of determination. Statistical measure to analize how good our model is predicting the real values y.
    :param y: Vector of "true" values.
    :param c: The coefficient matrix.
    :param x: Data matrix.
    :return: The coefficient of determination of the model.
    """
    yh, mean = innprod(c, x), y.mean()
    return ((yh-mean)**2).sum()/((y-mean)**2).sum()

