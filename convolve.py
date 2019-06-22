import numpy as np
from shift import *
def convolve(x, nu, boundary ):
    xconv = np.zeros(x.shape)
    s1 = int((nu.shape[0] - 1) / 2)
    s2 = int((nu.shape[1] - 1) / 2)
    for k in range(-s1, s1+1):
        for l in range(-s2, s2+1):
            xconv += nu[k+s1,l+s2]*shift(x,-k,-l,boundary)
    return xconv
