#!/usr/bin/env python
# coding: utf-8

import dill
import numpy as np
import bilby
import sys
from GMM_likelihood import GMMLikelihood
import random
from bilby.core.prior import Uniform, ConditionalUniform
G_c2_km_SM = 1.477
import pandas as pd
import json
import glob
import pickle
from tqdm import tqdm
import sys
from sampling_utils import UtilFuncs
from sampling_utils import UtilFuncs
import ILoveQ_utils
import argparse
import jax
jax.config.update("jax_enable_x64", True)


parser = argparse.ArgumentParser()
parser.add_argument('--gmm-file-list', '--fits', type=str, help="txt file with list of paths to gmm pickle files")
parser.add_argument('--n-samples-per-event', '-n', type=int, default=750, help="The number of per-event parameter samples that will be used for reweighting, divided by the number of events. For example, if you have 15 events and want to reweight a total of 1500 samples into the GMMs, you would set --n-samples-per-event 100.")
parser.add_argument("--backend", "-b", type=str, default="jax", help="Backend to use. Either numpy or jax")
parser.add_argument("--label", "-l", type=str, help="Label for the run")
parser.add_argument("--outdir", "-o", type=str, help="Output directory for the run", default="outdir")
parser.add_argument("--prior", "-p", type=str, help="Prior file for run")
parser.add_argument("--selection-samples", "-s", type=str, help="Path to file containing selection samples.")
parser.add_argument('--scale-chirp', type=float, default=1.1, help="Factor to multiply by the standard deviation of the chirp mass Gaussian fit when drawing the mass proposal samples. Larger number increases support in the tails.")
parser.add_argument("--eos", type=str, default=None, help="This determines whether/how to fit the EoS. Leave this empty to just fit masses. If this is 'UR' it will sample over UR model parameters.")
args = parser.parse_args()
                    
prior_file = args.prior
label = args.label
backend = args.backend
samples_per_event = args.n_samples_per_event
gmm_file_list = args.gmm_file_list
selection_samples_path = args.selection_samples
chirp_scale = args.scale_chirp
outdir = args.outdir
    
if backend == 'cupy':
    import cupy as xp
    util_funcs = UtilFuncs('cupy')
    run_with_cupy = True
elif backend == 'jax':
    import jax.numpy as xp
    import jax
    util_funcs = UtilFuncs('jax')
    run_with_cupy = True
else:
    xp = np
    util_funcs = UtilFuncs('numpy')
    run_with_cupy = False

prior = bilby.core.prior.PriorDict(prior_file)
print(prior.sample())
powerlaw = util_funcs.powerlaw

selection_samples = pd.read_csv(selection_samples_path)
Ninj = 10000
found_m1s = xp.array(selection_samples['mass_1_source'])
found_m2s = xp.array(selection_samples['mass_2_source'])


pdraws = xp.array(selection_samples['prior'])

def selection(params):
    weights = util_funcs.powerlaw(found_m1s, alpha = params['alpha'], low = params['mmin'], high = params['mpop']) * util_funcs.powerlaw(found_m2s, alpha = params['alpha'], low = params['mmin'], high = params['mpop']) / pdraws
    neff = xp.sum(weights)**2 / xp.sum(weights**2)
    return xp.where(neff > 4 * Likelihood.n_events, 1/Ninj * xp.sum( weights ), xp.inf)

print(f"Reading files from the list in {gmm_file_list}")
gw_gmm_files = np.loadtxt(gmm_file_list, dtype=str)
results = []
for file in tqdm(gw_gmm_files):
    with open(file, "rb") as ff:
        dat = pickle.load(ff)
        results.append(dat.result)

medians = xp.array([np.median(res.posterior['chirp_mass_source']) for res in results])
stdevs = xp.array([np.std(res.posterior['chirp_mass_source']) for res in results])
medians = medians[:,None]
stdevs = stdevs[:,None] * chirp_scale

n_pop_samples=len(results) * samples_per_event
chirp_samples = xp.array(util_funcs.icdf_normal(np.random.uniform(size=n_pop_samples//len(results)), medians, stdevs).flatten())
q_samples = xp.array(util_funcs.icdf_powerlaw(np.random.uniform(size=len(chirp_samples)), alpha = 1, minimum = 0.125, maximum=1))
m1s, m2s = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(chirp_mass = chirp_samples, mass_ratio = q_samples)
samples=dict()
samples['mass_1_source'] = m1s
samples['mass_2_source'] = m2s
samples['chirp_mass_source'] = bilby.gw.conversion.component_masses_to_chirp_mass(m1s, m2s)
samples['mass_ratio'] = samples['mass_2_source'] / samples['mass_1_source']

for key in samples:
    samples[key] = xp.asarray(samples[key])
    
jacobian = xp.asarray(samples['chirp_mass_source'] / samples['mass_1_source']**2) #d(M_c,q)/d(m1,m2)
p_draw = xp.asarray(util_funcs.pdf_normal(samples['chirp_mass_source'], medians, stdevs, 1/len(results)) * util_funcs.powerlaw(samples['mass_ratio'], alpha = 1, low = 0.125, high=1))

def population_weights(N, params, medians=medians, stdevs=stdevs, N_results=len(results) ):
    p_m1_m2 = util_funcs.powerlaw(samples['mass_1_source'], alpha = params['alpha'], low = params['mmin'], high= params['mpop']) * util_funcs.powerlaw(samples['mass_2_source'], alpha = params['alpha'], low = params['mmin'], high= params['mpop'])
    samples['weights'] = p_m1_m2 / p_draw / jacobian
    return samples

def population_weights_UR(N, params, medians=medians, stdevs=stdevs, N_results=len(results)):
    new_samples = population_weights(N, params, medians=medians, stdevs=stdevs, N_results=len(results))
    mtov = ILoveQ_utils.MTOV([params['m'], params['b']], 1.89)
    
    new_samples['weights'] *= (mtov > params['mpop']) 
    new_samples['lambda_1'] = xp.nan_to_num(ILoveQ_utils.Lambda_of_m(new_samples['mass_1_source'], Im_coeffs=[params['m'], params['b']]))
    new_samples['lambda_2'] = xp.nan_to_num(ILoveQ_utils.Lambda_of_m(new_samples['mass_2_source'], Im_coeffs=[params['m'], params['b']])) 
    return new_samples

if not args.eos:
    sample_eos = False
    population_func = population_weights
elif args.eos.lower() == "ur":
    sample_eos = True
    population_func = population_weights_UR

    
Likelihood = GMMLikelihood(gmm_files= gw_gmm_files, 
                         population_sample_func = population_func, selection_function = selection, 
                        backend="jax", n_pop_samples=n_pop_samples)

class Like(bilby.core.likelihood.Likelihood):
    def __init__(self):
        self.parameters = dict()
        super(Like, self).__init__(dict())
    def log_likelihood(self):
        Likelihood.parameters.update(self.parameters)
        ll = Likelihood.log_likelihood(jax=True)
        return ll
like_obj = Like()

for _ in range(100):
    like_obj.parameters.update(prior.sample())
    print(like_obj.log_likelihood()) #just make sure it works

result = bilby.core.sampler.run_sampler(like_obj, prior, label = f'{label}_output', outdir = outdir, sampler = 'dynesty', use_ratio = False, sample= 'acceptance-walk',naccept = 10, nlive= 1000, dlogz=0.1, nsteps=300)