import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x,y_range=1,y_shift=0,x_shift=0):
    return (y_range / (1 + np.exp(-x + x_shift))) + y_shift

if __name__ == "__main__":

    Y_RANGE = 1
    Y_SHIFT = -0.5
    X_SHIFT = 0
    DOMAIN = (-10,10)

    sig_func = lambda _x: sigmoid(_x,Y_RANGE,Y_SHIFT,X_SHIFT)

    x = np.arange(*DOMAIN,0.1)
    y = [sig_func(_x) for _x in x]

    plt.plot(x,y)
    plt.show()

