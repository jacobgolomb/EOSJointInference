import numpy as np
import scipy

class UtilFuncs(object):
    def __init__(self,backend = 'numpy'):
        if backend == 'numpy':
            self.xp = np
            self.erfinv = scipy.special.erfinv
            
        elif backend == 'cupy':
            import cupy as xp
            self.xp = xp
            from cupyx.scipy.special import erf, erfinv
            self.erfinv = erfinv
            self.erf = erf
        elif backend == "jax":
            import jax.numpy as jnp
            import jax
            self.xp = jnp
            self.erf = jax.scipy.special.erf
            self.erfinv = jax.scipy.special.erfinv
            

    def pdf_normal(self,xx, mu, sigma, weights=1):
        pdfs = weights * (sigma * np.sqrt(2 * np.pi))**-1 * self.xp.exp(-0.5 * (xx - mu)**2 /sigma**2)
        return self.xp.sum(pdfs, axis=0)

    def icdf_normal(self,unit_samples, mu, sigma):
        icdfs = mu + sigma*np.sqrt(2) * self.erfinv(2 * unit_samples - 1)
        return icdfs

    def icdf_powerlaw(self,val, alpha, minimum, maximum):
        if alpha == -1:
            return minimum * self.xp.exp(val * self.xp.log(maximum / minimum))
        else:
            return (minimum ** (1 + alpha) + val *
                    (maximum ** (1 + alpha) - minimum ** (1 + alpha))) ** (1. / (1 + alpha))

    def p_pop_eos(self,mmax, eos):
        if mmax < eos.MTOV:
            return 1
        else: 
            return 0

    def powerlaw(self,xx, alpha, high, low):
        r"""
        Power-law probability

        .. math::
            p(x) = \frac{1 + \alpha}{x_\max^{1 + \alpha} - x_\min^{1 + \alpha}} x^\alpha

        Parameters
        ----------
        xx: float, array-like
            The abscissa values (:math:`x`)
        alpha: float, array-like
            The spectral index of the distribution (:math:`\alpha`)
        high: float, array-like
            The maximum of the distribution (:math:`x_\min`)
        low: float, array-like
            The minimum of the distribution (:math:`x_\max`)

        Returns
        -------
        prob: float, array-like
            The distribution evaluated at `xx`

        """
        if self.xp.any(self.xp.asarray(low) < 0):
            raise ValueError(f"Parameter low must be greater or equal zero, low={low}.")
        
        norm = self.xp.where(alpha == -1, 1 / self.xp.log(high / low), (1 + alpha) / (high ** (1 + alpha) - low ** (1 + alpha)))
        prob = self.xp.power(xx, alpha)
        prob *= norm
        prob *= (xx <= high) & (xx >= low)
        return prob