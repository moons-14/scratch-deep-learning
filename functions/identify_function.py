import numpy as np
import matplotlib.pylab as plt


def identify_function(x):
    return x


x = np.arange(-5.0, 5.0, 0.1)
y = identify_function(x)
plt.plot(x, y)
plt.ylim(-5.0, 5.0)
plt.show()
