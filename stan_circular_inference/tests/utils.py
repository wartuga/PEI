import numpy as np

def in_circular_interval(val, low, up):
    val = val % (2*np.pi)
    low = low % (2*np.pi)
    up = up % (2*np.pi)
    if low <= up:
        return low <= val <= up
    else:
        return val >= low or val <= up