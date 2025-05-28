#!/usr/bin/env python
# coding: utf-8

import numpy as np
import jax.numpy as jnp
import argparse
import jax
import bilby
from functools import partial
import ILoveQ_utils
import models
from radio_likelihoods import EMPulsarLikelihood
from GMM_likelihood import GMMLikelihood

ILoveQ_utils.set_backend("jax")
models.set_backend("jax")
def conditional_mu_2(reference_parameters, mu1):
    return dict(minimum = mu1, maximum = reference_parameters['maximum'])

"""
The setting of the prior could/should be in a separate file to be neater
"""
prior = bilby.core.prior.ConditionalPriorDict()
prior['mpop'] = bilby.core.prior.Uniform(minimum=1.8, maximum=3, name='mpop', latex_label=r'$m_{\rm pop}$', unit=None, boundary=None)
prior['mmin']=1
prior['frac1'] = bilby.core.prior.Uniform(minimum = 0, maximum = 1)
prior['mu1'] = bilby.core.prior.Uniform(minimum=1, maximum=2)
prior['sigma1'] = bilby.core.prior.Uniform(minimum = 0.05, maximum=1)
prior['sigma2'] = bilby.core.prior.Uniform(minimum = 0.05, maximum = 1)
prior['mu2'] = bilby.core.prior.ConditionalUniform(minimum = 1, maximum = 2.5, condition_func = conditional_mu_2)
prior['m'] = bilby.core.prior.Uniform(name='m', minimum = -2.1, maximum=-1)
prior['b'] = bilby.core.prior.Uniform(name='b', minimum = 2, maximum= 4)

parser = argparse.ArgumentParser(description='Run inference on NICER and radio data')
parser.add_argument('--pulsar-path', type=str, help='Path to the pulsar h5 data file')
parser.add_argument('--nicer-gmms-file-list', type=str, help='Path to text file containing the paths to the NICER GMMs')
parser.add_argument('--n-samples', type=int, default=5000, help='Number of samples to draw from the NICER GMMs')
parser.add_argument('--outdir', type=str, default='./outdir_EM/', help='Output directory for the results')
args = parser.parse_args()

radio_data_path = args.pulsar_path
NICER_paths = np.loadtxt(args.nicer_gmms_file_list, dtype=str)

class RadioandNICERLikelihood(bilby.Likelihood):
    def __init__(self,radio_data_path, nicer_gmms, fiducial_population_params, backend='jax', n_samples=5000):

        self.radio_likelihood = EMPulsarLikelihood(path=radio_data_path, backend=backend)
        self.n_samples = n_samples
        self.parameters = dict( mmin=None, mpop=None, mu1=None, mu2=None, sigma1=None, sigma2=None, frac1=None, m=None, b=None)
        self.fiducial_parameters = fiducial_population_params
        self.draw_proposal_pulsars()
        self.likelihoodNICER = GMMLikelihood(gmm_files= nicer_gmms, 
                         population_sample_func = lambda N, P: models.weight_pulsar_population(N, P, self.proposal_pulsars), selection_function = lambda x: 1, 
                        backend="jax", n_pop_samples=n_samples)
        self.xp = self.likelihoodNICER.xp
        super().__init__(parameters = self.parameters)
        
    def draw_proposal_pulsars(self):
        self.proposal_pulsars =  models.draw_fiducial_pulsar_samples(seed=1, **self.fiducial_parameters)
        self.proposal_radio_likelihoods = self.radio_likelihood.likelihood_per_event_per_sample(self.proposal_pulsars['mass'])
    
    def radio_log_likelihood(self, samples):
        return self.pulsar_object.log_likelihood(samples)
    
    def NICER_log_likelihood(self):
        self.likelihoodNICER.parameters.update(self.parameters)
        llNICER = self.likelihoodNICER.log_likelihood()
        return llNICER
    
    #@partial(jax.jit, static_argnums=(0,))
    def pulsar_log_likelihood(self, mass_weights):
        return self.xp.sum(self.xp.log(
            self.xp.mean(
                self.xp.exp(
                    self.xp.log(self.proposal_radio_likelihoods) + self.xp.log(mass_weights)), axis=-1)
        )
                               )
        
    
    def log_likelihood(self):
        llNICER = self.NICER_log_likelihood()
        mass_weights = self.likelihoodNICER.population_samples['weights']
        llPulsars = self.pulsar_log_likelihood(mass_weights)
        return llNICER + llPulsars


likelihood = RadioandNICERLikelihood(radio_data_path=radio_data_path,
                                    nicer_gmms=NICER_paths, n_samples=5000, fiducial_population_params={'mmin': 1, 'mmax': 3.2, 'mu': 1.35, 'sigma': 0.05, 'gaussian_frac': 0.4})

testparams = {'mmin': 1, 'mpop': 2.5, 'mu1': 1.35, 'sigma1': 0.05, 'mu2': 1.7, 'sigma2': 0.1, 'frac1': 0.2, 'm':-1.1, 'b': 3.0}
result =  bilby.core.sampler.run_sampler(likelihood, prior, label = 'NICER_radio_UR', outdir = args.outdir, sampler = 'dynesty')