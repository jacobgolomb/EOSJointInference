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
import gc
import glob
from tqdm import tqdm, trange
import argparse
import h5py
import sys
from sampling_utils import UtilFuncs
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gmm-files', '--fits', type=str, help="txt file with list of paths to gmm pickle files")
parser.add_argument("--samples-file", "-s", type=str, help="h5 file with single event samples")
parser.add_argument("--output-directory", "-o", type=str, help="directory to save likelihoods.")
parser.add_argument("--backend", "-b", type=str, default="jax", help="Backend to use. Either numpy or jax")


args = parser.parse_args()
gmm_files = args.gmm_files
samples_file = args.samples_file.split(".h5")[0]+".h5"
output_dir = args.output_directory
backend = args.backend

if os.path.isdir(output_dir):
    if os.listdir(output_dir) is not None:
        print("Output directory is not empty. This may overwrite files that are present!")
else:
    print(f"Creating directory {output_dir}")
    os.makedirs(output_dir)
    
if backend.lower() == 'cupy':
    import cupy as xp
    util_funcs = UtilFuncs('cupy')
elif backend.lower() == 'jax':
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

gmm_files = np.loadtxt(gmm_files, dtype=str)
results = []
for file in gmm_files:
    res = pickle.load(open(file, "rb")).result
    results.append(res)
    

with h5py.File(samples_file) as population_samples:
    mass_samples = xp.asarray(population_samples["mass"][()])
    pdraw_population_samples = xp.asarray(population_samples['p_draw'][()]) 
    compactness = {eos: xp.asarray(population_samples[eos]).T for eos in population_samples.keys() if "eos" in eos}
    eos_keys = [key for key in population_samples.keys() if 'eos_' in key]
    
pop_samples_per_event = len(pdraw_population_samples) // len(results)
print(f"Assuming {pop_samples_per_event} samples per event")

samples={}    
samples['mass'] = mass_samples

def sample_masses_and_compactness(N, params, eos):
    samples['compactness'] = compactness[eos]
    return samples

def dummy_selection(params):
    return 1

full_likelihood = GMMLikelihood(gmm_files= gmm_files, 
                         population_sample_func = sample_masses_and_compactness, selection_function = dummy_selection, 
                        backend = "jax", n_pop_samples=len(results) * pop_samples_per_event)

full_likelihood.parameters.update(dict())
likelihoods = {}
likelihoods_temp = {}
for ii, eos in enumerate(tqdm(eos_keys)):
    new_likelihood = jax.block_until_ready(full_likelihood.log_likelihood({'eos': eos}, jax=True))
    likelihoods_temp[eos] = full_likelihood.log_likelihood_per_event_per_sample
    if ii> 0 and ii % 500 == 0:
        for eos_temp in likelihoods_temp:
            likelihoods[eos_temp] = np.asarray(likelihoods_temp[eos_temp])
        print(f"Transferring to CPU at {ii}")
        with open(f"{output_dir}/likelihoods_{ii}.pkl", "wb") as ww:
            pickle.dump(likelihoods, ww)
        print("Done saving")
        likelihoods_temp = {}
        likelihoods = {}


likelihoods.update({eos: np.asarray(likelihoods_temp[eos]) for eos in likelihoods_temp})
with open(f"{output_dir}/likelihoods_{ii}.pkl", "wb") as ww:
    pickle.dump(likelihoods, ww)
