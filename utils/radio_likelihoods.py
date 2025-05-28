import jax.numpy as jnp
import jax
import scipy
import numpy as np
from bilby.core.prior import TruncatedGaussian
import h5py

class GGpulsar(object):
    def __init__(self, mu_p, sigma_p, xp = "jax"):
        if xp == "jax":
            self.xp=jnp
            self.norm = jax.scipy.stats.norm
        elif (xp == "np") or (xp == "numpy"):
            self.xp = np
            self.norm = scipy.stats.norm

        self.mu_p = self.xp.asarray(mu_p)[...,None]
        self.sigma_p = self.xp.asarray(sigma_p)[...,None]
        
    def likelihood_m(self, m):
        prob = self.norm.pdf(x=m,loc=self.mu_p, scale=self.sigma_p)
        return prob
    
class QGpulsar(object):
    def __init__(self, mu_q, sigma_q, f, n_samples = 2000, xp = "jax"):
        self.n_obs = len(mu_q)
        self.mu_q = mu_q
        self.sigma_q = sigma_q
        if xp == "jax":
            self.xp=jnp
        elif (xp == "np") or (xp == "numpy"):
            self.xp = np
            
        self.f = self.xp.asarray(f)
        self.n_samples = n_samples
        self.draw_q_samples()
        
    def draw_q_samples(self):
        q_samples = []
        for i in range(len(self.mu_q)):
            pq = TruncatedGaussian(mu = self.mu_q[i], sigma = self.sigma_q[i], minimum = 0, maximum = np.inf)
            q_samples.append(pq.sample(self.n_samples))
        self.q_samples = self.xp.array(q_samples)
        
    def likelihood_m(self, m):
        q = self.q_samples
        f = self.f
        # Reshape m and f to broadcast properly with q
        m = m[:, None, None]  # Shape (len(m), 1, 1)
        f = f[None, :, None]  # Shape (1, n_obs, 1)
        q = q[None, :, :]  # Shape (1, n_obs, n_samples)

        term1 = (1 + q) ** (4 / 3)
        term2 = (f / m) ** (2 / 3) * term1 / q ** 2
        weights = term1 / (3 * f ** (1 / 3) * m ** (2 / 3) * q ** 2 * self.xp.sqrt(1 - term2))
        weights *= m

        return self.xp.mean(weights, axis=2).T  # Return mean over samples axis, shape (n_obs, len(m))
    
class MTpulsar(object):
    def __init__(self, mu_mt, sigma_mt, f, n_samples = 3000, min_mc = 0.2, xp = np):
        self.n_obs = len(mu_mt)
        self.mu_mt = mu_mt
        self.sigma_mt = sigma_mt
        if xp == "jax":
            self.xp=jnp
            self.norm = jax.scipy.stats.norm
        elif (xp == "np") or (xp == "numpy"):
            self.xp = np
            self.norm = scipy.stats.norm
        
        self.f = self.xp.array(f[...,None])
        self.n_samples = n_samples
        self.min_mc = min_mc
        self.draw_mt_samples()
        
    def draw_mt_samples(self):
        mt_samples = []
        for i in range(len(self.mu_mt)):
            pmt = TruncatedGaussian(mu = float(self.mu_mt[i]), sigma = float(self.sigma_mt[i]), minimum = 0, maximum = 10)
            samples = pmt.sample(self.n_samples)
            mt_samples.append(samples)
            
        self.mt_samples = self.xp.asarray(mt_samples).T[...,None,None]
        
    def likelihood_m(self, m):
        #mt = self.mt_samples
        weight_array = []
        for i in range(len(self.f)):
            f = self.f[i]
            pmt = TruncatedGaussian(mu = float(self.mu_mt[i]), sigma = float(self.sigma_mt[i]), minimum = 0, maximum = np.inf)
            mc = pmt.sample(10000)[...,None] - m
            
            mt = mc + m
            q = mc/m
            term1 = (f/m)**(2/3) * (1 + q)**(4/3) / q**2
            term1 = jnp.where(term1 < 1, term1, 0)
            weights= ((1 + q)**(4/3) / (3 * f**(1/3) * (m**(2/3) * q**2) * 
                                  self.xp.sqrt(1 - term1)
                     ) 
                     )
            weights = jnp.where(term1 == 0, 0, weights)
            weights = jnp.where(mc > 0, weights, 0)
            weights = self.xp.mean(weights, axis=0)
            
            weight_array.append(weights)
        return self.xp.asarray(weight_array)

class EMPulsarLikelihood(object):
    def __init__(self,path, backend, use_cache=True):
        data = h5py.File(path)
        if backend == "jax":
            self.xp=jnp
            self.norm = jax.scipy.stats.norm
        elif (backend == "np") or (backend == "numpy"):
            self.xp = np
            self.norm = scipy.stats.norm
        mu_ps = self.xp.array(data['mp']['mp_mean'][()])
        sigma_ps = self.xp.array(data['mp']['mp_std'][()])
        self.GP = GGpulsar(mu_ps, sigma_ps, xp = backend)
        
        mu_qs = self.xp.array(data['q']['q_mean'][()])
        sigma_qs = self.xp.array(data['q']['q_std'][()])
        q_fs = self.xp.array(data['q']['f'][()])
        self.GQ = QGpulsar(mu_qs, sigma_qs, q_fs, xp = backend)

        mu_mts = self.xp.array(data['mt']['mt_mean'][()])
        sigma_mts = self.xp.array(data['mt']['mt_std'][()])
        mt_fs = self.xp.array(data['mt']['f'][()])
        self.GMT = MTpulsar(mu_mts, sigma_mts, mt_fs, xp = backend)
        data.close()
        
    def log_likelihood_mc(self, m):
        per_event_GMTs = self.xp.mean(self.GMT.likelihood_m(m), axis=-1)
        per_event_GQs = self.xp.mean(self.GQ.likelihood_m(m), axis=-1)
        per_event_GPs = self.xp.mean(self.GP.likelihood_m(m), axis=-1)
      
        return self.xp.sum(self.xp.log(per_event_GMTs)) + self.xp.sum(self.xp.log(per_event_GPs)) + self.xp.sum(self.xp.log(per_event_GQs))
    
    def log_likelihood_interp(self, m):
        per_event_GMTs = self.xp.mean(self.GMT.likelihood_m_cached(m), axis=-1)
        per_event_GQs = self.xp.mean(self.GQ.likelihood_m_cached(m), axis=-1)
        per_event_GPs = self.xp.mean(self.GP.likelihood_m(m), axis=-1)

        return self.xp.sum(self.xp.log(per_event_GMTs)) + self.xp.sum(self.xp.log(per_event_GPs)) + self.xp.sum(self.xp.log(per_event_GQs))
    
    def log_likelihood(self, m):
        
        return self.log_likelihood_func(m)
    
    def likelihood_per_event_per_sample(self, m):
        per_event_per_sample_GMTs = self.GMT.likelihood_m(m)
        per_event_per_sample_GQs = self.GQ.likelihood_m(m)
        per_event_per_sample_GPs = self.GP.likelihood_m(m)
        return self.xp.concatenate((per_event_per_sample_GMTs,per_event_per_sample_GQs,per_event_per_sample_GPs))
        