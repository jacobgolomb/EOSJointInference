from bilby.core.likelihood import Likelihood
from bilby.core.utils import logger
from bilby.core.prior import Uniform, Interped, ConditionalUniform
import numpy as np
from sklearn.mixture import GaussianMixture
import copy 
import corner 
import dill
from scipy.stats import norm 
import jax
from functools import partial
from tqdm import tqdm
from GMM_likelihood import CDFs

class GMMLikelihood(object):
    
    def __init__(
        self, gmm_files, population_sample_func, ln_evidences=None, selection_function=lambda args: 1, backend="numpy",
        n_pop_samples=15000, outdir='gmm_outdir', label='gmmlikelihood', generate_source_frame=None, n_components=None, **kwargs
    ):
        if backend == "numpy":
            print("using numpy")
            from scipy.special import erfinv
            self.xp = np
            self.erfinv = erfinv
            self.multivariate_logpdf_func = multivariate_normal_logpdf
            self.cdf = self.cdf_regular
        elif backend == "jax":
            print("using jax")
            import jax
            import jax.numpy as jnp
            import jax.scipy.special as jss
            self.xp = jnp
            self.erfinv = jss.erfinv
            self.multivariate_logpdf_func = jax.jit( multivariate_normal_logpdf, static_argnames=("xp",))
            self.cdf = self.cdf_jax
            self.jit = jax.jit
        CDFs.set_backend(backend)
        
        self.backend = backend


        self.population_sample_func = population_sample_func
        Likelihood.__init__(self, parameters=dict())
        
        if ln_evidences is not None:
            self.total_noise_evidence = self.xp.sum(ln_evidences)
        else:
            self.total_noise_evidence = self.xp.nan

        self.selection_function = selection_function
        self.n_pop_samples = n_pop_samples
        self.outdir = outdir
        self.label = label
        self.gmm_files = gmm_files
        self.load_and_init_gmms()

        self.n_events = len(self.gmms)
        self.gmm_parameters = self.gmms[0].parameters
        self.setup_prior_map()
        self.dims = len(self.gmm_parameters)

        
    def cdf_func(self, samples, priors):  
        cdfs = self.transformer.cdf(samples, priors)
        return cdfs
    """
    def transformation_function(self, samples, priors, n_samples):
        posts_transformed = self.transformer.transformation_function(samples, priors, n_samples)
        return posts_transformed
    """
    def load_and_init_gmms(self):
        print(f"Loading {len(self.gmm_files)} GMM files.")
        gmms = []
        for file in tqdm(self.gmm_files):
            with open(file, 'rb') as gg:
                gmm_here = dill.load(gg)
            gmms.append(gmm_here)

        self.gmms = sorted(gmms, key=lambda x: x.label)

        idx_max = np.argmax([gmm.n_components for gmm in self.gmms])

        means = []
        covs = []
        weights = []

        for gmm in self.gmms:
            means.append(np.resize(gmm.gmm.means_ , self.gmms[idx_max].gmm.means_.shape))
            covs.append(np.resize(gmm.gmm.covariances_ , self.gmms[idx_max].gmm.covariances_.shape))
            weights.append(np.pad(gmm.gmm.weights_ , (0, self.gmms[idx_max].gmm.n_components - len(gmm.gmm.weights_))))
        covs = self.xp.array(covs)
        weights = self.xp.array(weights)
        self.means = self.xp.array(means)
        self.invcovs = self.xp.linalg.inv(covs)
        self.covs = self.xp.array(covs)
        self.detcovs = self.xp.linalg.det(covs)[..., None]
        self.detinvcovs = self.xp.linalg.det(self.invcovs)[..., None]
        self.weights = self.xp.asarray(weights)[..., None]
        
    def log_likelihood(self, pop_kwargs={}, jax=True):
        self.population_samples = self.population_sample_func(self.n_pop_samples, self.parameters, **pop_kwargs)
        extra_weights = self.population_samples.get("weights", 1)
        if not self.population_samples:
            return -self.xp.inf
        transformed_samples = self.transformation_function(self.population_samples)
        self.transformed_samples = transformed_samples
        
        ln_l = self.log_likelihood_samples(transformed_samples, extra_weights)
        
        ln_l += self._get_selection_factor()
        return ln_l
        """
        if self.xp.isnan(ln_l):
            return -self.xp.nan_to_num(self.xp.inf)
        else:
            return self.xp.nan_to_num(ln_l)
        """
       
    def log_likelihood_samples(self, transformed_samples, extra_weights = 1):

        xx = self.xp.moveaxis(transformed_samples, -1, -2) #shape is now (n_events, n_pop_samples, n_dims)

        log_pdfs = self.multivariate_logpdf_func(xx, self.means, self.invcovs, self.detcovs, self.weights, self.xp)
        log_prior_norm = self.xp.sum(self.log_normalpdf(xx), axis=-1)

        log_likelihood_per_event_per_sample = self.xp.nan_to_num(log_pdfs - log_prior_norm, nan=-self.xp.inf, posinf=-self.xp.inf, neginf=-self.xp.inf) + self.xp.log(extra_weights)
        self.log_likelihood_per_event_per_sample = log_likelihood_per_event_per_sample
        
        weight_per_event_per_sample = self.xp.exp(log_likelihood_per_event_per_sample)
        likelihood_per_event = self.xp.sum(weight_per_event_per_sample, axis = -1) / self.n_pop_samples

        #self.per_event_log_likelihood = self.xp.log(likelihood_per_event)
        loglike = self.xp.sum(self.xp.log(likelihood_per_event))
        return loglike
        
    def normalpdf(self, arr):
        return (1/self.xp.sqrt(2*np.pi)) * self.xp.exp(-(arr**2)/2)

    def log_normalpdf(self, arr):
        return -0.5 * self.xp.log(2*np.pi) - (arr**2)/2 
    
    def log_likelihood_CPU(self, transformed_samples):

        likes = []
        for i in range(len(self.gmms)):
            samples_prime = np.squeeze(transformed_samples[i])
            prior_norm = np.prod(norm.pdf(samples_prime), axis=1)
            likes.append(np.sum(np.nan_to_num(np.exp(self.gmms[i].score_samples(transformed_samples[i]))/prior_norm)))
        loglike = np.sum(np.log(likes)- np.log(self.true_n_pop_samples))
        
        return loglike
    
    def _get_selection_factor(self):
        return -self.n_events * self.xp.log(self.selection_function(self.parameters))
    
    
    def setup_prior_map(self):     
    
        prior_map = dict()

        for key in self.gmm_parameters:
            prior_map[key] = []
            event_priors = np.array([event.transformer.priors[key] for event in self.gmms])
            seen = list()
            for prior in event_priors:
                if prior.cdf_func.__name__ not in seen:
                    seen.append(prior.cdf_func.__name__)
                    occurences = np.where(np.array([pr.cdf_func.__name__ for pr in event_priors]) == prior.cdf_func.__name__)[0]
                    cdf_parameters = {key: self.xp.array([event_priors[occ].cdf_parameters[key] for occ in occurences])[...,None] for key in prior.cdf_parameters.keys() if key not in ['xx', 'YY', 'xp', 'fp']}
                    if prior.cdf_func.__name__ == "interpedCDF":
                        cdf_parameters['xps'] = self.xp.array([self.xp.linspace(min(event.cdf_parameters['xps']), max(event.cdf_parameters['xps']), 10000) for event in event_priors[occurences]])
                        cdf_parameters['fps'] = self.xp.array([self.xp.interp(x=cdf_parameters['xps'][ii], xp=event_priors[ii].cdf_parameters['xps'], fp=event_priors[ii].cdf_parameters['fps'], left=0.0, right=1.0) for ii in occurences])

                        interp_func = self.xp.vectorize(self.xp.interp,signature="(n),(m),(m)->(n)", excluded=["left", "right"])
                        cdffunc = lambda vals, xps, fps: interp_func(vals, xps, fps, left=0., right=1.)
                        if self.backend == "jax":
                            cdffunc = self.jit(cdffunc)
                    else:
                        cdffunc = partial(prior.cdf_func, **cdf_parameters)
                    prior_map[key].append({'cdf_func': cdffunc, 'occurences': occurences, 'cdf_parameters': cdf_parameters})
        self.prior_map = prior_map

    def cdf_regular(self, samples):
        n_samples = samples[list(samples.keys())[0]].shape
        cdfs = self.xp.zeros((self.n_events, self.dims, *n_samples))
        for i, key in enumerate(self.gmm_parameters):
            for element in self.prior_map[key]:
                indices = element['occurences']
                cdfs[indices, i] = self.gmms[indices[0]].transformer.cdf_single_dimension(samples, key)
        return cdfs
    
    def cdf_jax(self, samples):
        n_samples = samples[list(samples.keys())[0]].shape
        cdfs = self.xp.zeros((self.n_events, self.dims, *n_samples))
        for i, key in enumerate(self.gmm_parameters):
            for element in self.prior_map[key]:
                indices = element['occurences']
                cdfs = cdfs.at[indices, i].set(element['cdf_func'](samples[key], **element['cdf_parameters']))
        return cdfs
    
    
    def transformation_function(self, samples):

        cdf = self.cdf(samples)
        posteriors_transformed = self.xp.nan_to_num(self.xp.sqrt(2)*self.erfinv(2*cdf - 1))
        return posteriors_transformed

def multivariate_normal_logpdf(xx, mu, inverse_cov, det_cov, weights, xp):
    """
    xx: points to evaluate (npoints, nevents, ndim)
    mu: mean vector (nevents, ncomponents, ndim)
    where ncomponents is number of Gaussians in each GMM
    inverse_cov: inverse of the covariance matrix (nevents, ncomponents, ndim, ndim)
    det_cov: determinant of the inverse of the covariance matrix (nevents, ncomponents)
    weights: weights of the components (nevents, ncomponents)

    Returns
    -------------
    logpdf: log probability density function (npoints, nevents)
    """
    ndim = mu.shape[-1]

    xminusmu = xp.expand_dims(xx, -3) - xp.expand_dims(mu, -2)

    #logexp = - 1/2 * xp.einsum('ijkl, ijlm, ijkm -> ijk', xminusmu, inverse_cov, xminusmu, optimize=True)
    x_inv_cov = xp.einsum('...jk,...kl->...jl', xminusmu, inverse_cov)  # (npoints, nevents, ncomponents, ndim)
    logexp = -0.5 * xp.einsum('...ij,...ij->...i', x_inv_cov, xminusmu)  # (npoints, nevents, ncomponents)
   
    log_prefactor = -ndim/2 * xp.log(2*np.pi) - 1/2 * xp.log(det_cov)
    log_pdf_per_component = log_prefactor + logexp 
    log_pdf = xp.log(xp.sum(xp.exp(log_pdf_per_component + xp.log(weights)), axis= 1))
    return log_pdf   
