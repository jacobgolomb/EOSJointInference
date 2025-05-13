#!/usr/bin/env python
# coding: utf-8

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import bilby
import sys
from functools import partial
sys.path.append('../../utils/')
sys.path.append('../')
import ILoveQ_utils
import models
from radio_likelihoods import EMPulsarLikelihood
from GMM_likelihood import GMMLikelihood
from priors import prior
import numpyro
import numpyro.distributions as dist
from numpyro_utils import get_priors_from_file, sample_parameters_from_dict
from numpyro.infer import NUTS, MCMC
ILoveQ_utils.set_backend("jax")
models.set_backend("jax")
paths = np.loadtxt("NICEREvents.txt", dtype=str)


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
    
    @partial(jax.jit, static_argnums=(0,))
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


likelihood = RadioandNICERLikelihood(radio_data_path='../../AlsingNSMassReplication/pulsars_noNICER.h5',
                                    nicer_gmms=paths, n_samples=5000, fiducial_population_params={'mmin': 1, 'mmax': 3.2, 'mu': 1.35, 'sigma': 0.05, 'gaussian_frac': 0.4})

testparams = {'mmin': 1, 'mpop': 2.5, 'mu1': 1.35, 'sigma1': 0.05, 'mu2': 1.7, 'sigma2': 0.1, 'frac1': 0.2, 'm':-1.1, 'b': 3.0}
result =  bilby.core.sampler.run_sampler(likelihood, prior, label = 'NICER_radio_UR', outdir = './outdir/', sampler = 'dynesty')