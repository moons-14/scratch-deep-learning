import os
import sys

sys.path.append(os.pardir)

import numpy as np

from functions.cross_entropy_error import cross_entropy_error
from functions.softmax_function import softmax_function


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # ガウス分布で初期化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax_function(z)
        loss = cross_entropy_error(y, t)

        return loss
