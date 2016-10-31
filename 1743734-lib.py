import numpy as np


def innprod(x, y):
    if x.shape[-1] == y.shape[0]:
        return (x * y).sum(axis=-1)
    else:
        return np.nan


def grad(y, c, x):
    return -2.0*innprod(c.T, y - innprod(c, x))


def rss(y, x, w):
    a = y - innprod(x, w)
    return innprod(a.T, a)


def descent(y, x, alpha=1e-3, itr=1e2, eps=1e-6):
    w = np.zeros(x[0].size)
    while itr > 0:
        w1 = w - alpha * grad(y, x, w)
        if np.linalg.norm(w1 - w, 1)/np.linalg.norm(w, 1) < eps:
            return w1
        w = w1
        itr -= 1
    return w


def r2(y, c, x):
    yh, mean = innprod(c, x), y.mean()
    return ((yh-mean)**2).sum()/((y-mean)**2).sum()

