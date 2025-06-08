try:
    import cupy as cp
    from cupyx.scipy.special import erf as cperf
except ImportError:
    pass

import numpy as np
from scipy.special import erf as scipyerf
xp = np
def set_backend(backend):
    global xp, erf
    if backend=="numpy":
        xp = np
        erf = scipyerf
        print("setting backend to numpy")
    elif backend == "jax":
        import jax
        xp = jax.numpy
        erf = jax.scipy.special.erf
        print("setting backend to jax")


def uniformCDF(val, minimum, maximum):
    _cdf = (val - minimum) / (maximum - minimum)
    _cdf = xp.minimum(_cdf, 1)
    _cdf = xp.maximum(_cdf, 0)
    return _cdf

def interpedCDF(val, xps, fps):

    return xp.interp(val, xps, fps, left=0., right=1.)

def powerlawCDF(val, alpha, minimum, maximum):
    _cdf = xp.where(alpha==-1, (xp.log(val / minimum)/xp.log(maximum / minimum)), xp.atleast_1d(val ** (alpha + 1) - minimum ** (alpha + 1)) /(maximum ** (alpha + 1) - minimum ** (alpha + 1)))
        
    _cdf = xp.minimum(_cdf, 1)
    _cdf = xp.maximum(_cdf, 0)
    return _cdf

def gaussianCDF(val, mu, sigma):
    return (1 - erf((mu - val) / 2 ** 0.5 / sigma)) / 2