import numpy as np

def in_circular_interval(val, low, up):
    val = val % (2*np.pi)
    low = low % (2*np.pi)
    up = up % (2*np.pi)
    if low <= up:
        return low <= val <= up
    else:
        return val >= low or val <= up
    
def in_circular_interval(val, low, up):
    return (val > low and val < up or val > up and val < (low + (2*np.pi)) or val < low and val > (up - (2*np.pi)))

def correct_inferred_values(mu, low_mu, up_mu, var, low_var, up_var):
    return in_circular_interval(mu, low_mu, up_mu) and var > low_var and var < up_var