#!/usr/bin/env python
# coding: utf-8

import dill
import numpy as np
import bilby
import sys
from GMM_likelihood import GMMLikelihood
import random
from bilby.core.prior import Uniform, ConditionalUniform
from bilby.gw.conversion import component_masses_to_chirp_mass
G_c2_km_SM = 1.477
import pickle
import pandas as pd
import json
import glob
from tqdm import tqdm, trange
import argparse
import h5py
import sys
from sampling_utils import UtilFuncs


parser = argparse.ArgumentParser()
parser.add_argument('--log-likelihood-files', '-l', type=str, help="Files to glob for the log likelihoods")
parser.add_argument('--proposal-result', '-p', type=str, help="Path to result file for the proposal hyperparameter")
parser.add_argument('--selection-samples', '-s', type=str, help="Path to file with selection samples")
parser.add_argument('--samples-from-population', '-sp', type=str, help="Path to h5 file with the fiducial samples from the population (i.e. masses and lambdas)")
parser.add_argument('--gw-gmm-files', '--fits', '-f', type=str, help="Path to list of gmm fits")
parser.add_argument('--output', '-o', type=str, help="File to save output", default="samples_with_weights.pkl")
parser.add_argument('--backend', '-b', type=str, help="Backend, either numpy or jax", default='jax') 
args=parser.parse_args()

backend =args.backend
if backend.lower() == 'jax':
    import jax
    use_jax=True
    xp = jax.numpy
    util_funcs = UtilFuncs('jax')
    jax.device = jax.devices("gpu")[0]
    jax.config.update("jax_enable_x64", True)

elif backend.lower() == 'numpy':
    xp = np
    use_jax=False
    util_funcs = UtilFuncs('numpy')
    

output_file = args.output.split(".pkl")[0]+".pkl"

log_likelihood_files = glob.glob(args.log_likelihood_files)

proposal_result = bilby.result.read_in_result(args.proposal_result)
powerlaw = util_funcs.powerlaw
mass_params = proposal_result.posterior


n_pop_hyper_samples = len(mass_params)
mass_params = {key: xp.array(mass_params[key]).reshape((-1,1)) for key in mass_params}
selection_samples = pd.read_csv(args.selection_samples)
Ninj = 10000
found_m1s = xp.array(selection_samples['mass_1_source'])
found_m2s = xp.array(selection_samples['mass_2_source'])
pdraws = xp.array(selection_samples['prior'])

gw_gmm_files = np.loadtxt(args.gw_gmm_files, dtype=str)
n_events = len(gw_gmm_files)
def selection_function(params):
    weights = util_funcs.powerlaw(found_m1s, alpha = params['alpha'], low = params['mmin'], high = params['mpop']) * util_funcs.powerlaw(found_m2s, alpha = params['alpha'], low = params['mmin'], high = params['mpop']) / pdraws
    neff = xp.sum(weights, axis=1)**2 / xp.sum(weights**2, axis=1)
    return xp.where(neff > 4 *n_events, 1/Ninj * xp.sum( weights , axis=1), xp.inf)

population_samples = h5py.File(args.samples_from_population)

mass_samples = xp.asarray(population_samples["masses"][()])
m1s, m2s = mass_samples.T
pdraw_population_samples = xp.asarray(population_samples['p_draw'][()]) * xp.asarray(population_samples['jacobian_d(M_c,q)/d(m1,m2)'][()])
#lambdas = {eos: xp.asarray(population_samples[eos]).T for eos in population_samples.keys() if "eos" in eos}

def get_pop_weight(params):
    p_m1_m2 = util_funcs.powerlaw(m1s, alpha = params['alpha'], low = params['mmin'], high= params['mpop']) * util_funcs.powerlaw(m2s, alpha = params['alpha'], low = params['mmin'], high= params['mpop'])
    weights =p_m1_m2 / pdraw_population_samples
    return weights

@jax.jit
def calc_log_like(mass_pop_weights, log_likelihood):
    like_term = log_likelihood + xp.log(mass_pop_weights)[:,None,:]
    per_event = xp.mean(xp.exp(like_term), axis=-1)
    return xp.sum(xp.log(per_event), axis=-1) 

if __name__ == "__main__":
    samples_with_weights = {}

    pop_weights = get_pop_weight(mass_params)

    selection = selection_function(mass_params)
    samples_with_weights["mass_samples"] = {key: np.array(mass_params[key]).reshape((-1,1)) for key in mass_params}

    batch = 200
    while n_pop_hyper_samples % batch > 0:
        batch += 1
    print(f"using a batch size of {batch}")

    for kk, file in enumerate(log_likelihood_files):
        print(f"Loading file {kk}/{len(log_likelihood_files)}")
        with open(file, "rb") as rr:
            log_likelihoods = pickle.load(rr)
        print("Done loading file")
        eos_keys = [key for key in log_likelihoods.keys() if 'eos_' in key]
        for eos in tqdm(eos_keys):
            loglike_here = xp.asarray(log_likelihoods[eos])

            loglikes = np.zeros(n_pop_hyper_samples)
            for jj in range(n_pop_hyper_samples//batch):
                pop_weights_here = pop_weights[jj*batch: (jj+1)*batch]
                loglikes[jj*batch: (jj+1)*batch] = np.asarray(calc_log_like(pop_weights_here, loglike_here))
            samples_with_weights[eos] = loglikes - n_events * np.log(selection)

    with open(output_file, "wb") as ww:
        pickle.dump(samples_with_weights, ww)