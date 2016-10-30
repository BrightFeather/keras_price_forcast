import numpy as np

def R_square(y, f):
    tot = np.sum((y - y.mean())**2)
    res = np.sum((y - f)**2)
    print(res, tot)
    return 1 - (res/tot)