import numpy as np
import matplotlib.pyplot as plt


def gendata(w, n):
    d = len(w)
    x = np.ones((n, d))
    x[:, 1:] = 10.0 * np.random.rand(n, d - 1)
    y = innprod(x, w) + np.random.normal(size=n, loc=0, scale=1)

    return (x, y)


def innprod(x, y):
    if x.shape[-1] == y.shape[0]:
        return (x * y).sum(axis=-1)
    else:
        return np.nan


def grad(y, c, x):
    return -2 * innprod(c.T, y - innprod(c, x))


def rss(y, x, w):
    a = y - innprod(x, w)
    return innprod(a.T, a)


def descent(y, x, alpha, itr):
    w = np.zeros(x[0].size)
    # Gradient figure
    plt.figure()
    plt.plot(x[:, 1], y, 'ro')
    rssval = []
    rssval.append(rss(y, x, w))
    while itr >= 0:
        w = w - alpha * grad(y, x, w)
        rssval.append(rss(y, x, w))
        plt.plot([0, 10], [w[0], w[1] * 10 + w[0]], 'k-', lw=0.5)
        itr -= 1
    plt.show()

    plt.figure()
    plt.plot(range(len(rssval)), rssval, 'ro')
    plt.show()
    print(rssval)
    return w


def genplot(x, y, a, b):
    plt.figure()
    plt.plot([0, 10], [b, a * 10 + b], 'k-', lw=2)
    plt.plot(x[:, 1], y, 'ro')
    plt.show()

