import numpy as np


def numerical_gradient(f, X):
    h = 1e-4
    grad = np.zeros_like(X)

    it = np.nditer(X, flags=["multi_index"], op_flags=[["readwrite"]])
    while not it.finished:
        idx = it.multi_index
        tmp_val = X[idx]
        X[idx] = float(tmp_val) + h
        fxh1 = f(X)  # f(x + h)

        X[idx] = tmp_val - h
        fxh2 = f(X)  # f(x - h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        X[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad
