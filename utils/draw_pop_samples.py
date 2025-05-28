#!/usr/bin/env python

import dill
import matplotlib.pyplot as plt
import numpy as np
import bilby
import sys
import random
from bilby.core.prior import Uniform, ConditionalUniform
G_c2_km_SM = 1.477
import pandas as pd
import json
import glob
import h5py
from tqdm import tqdm
import sys
from sampling_utils import UtilFuncs
sys.path.append('/home/isaac.legred/EoSPopulation/Reweighting/')
from branched_interpolator import choose_macro_per_m
import pickle
import argparse

xp = np
util_funcs = UtilFuncs('numpy')

parser = argparse.ArgumentParser()
parser.add_argument('--gw-gmm-files', '--fits', type=str, help="txt file with list of paths to gmm pickle files")
parser.add_argument('--n-samples-per-event', '-n', type=int, default=750, help="The number of per-event parameter samples that will be used for reweighting, divided by the number of events. For example, if you have 15 events and want to reweight a total of 1500 samples into the GMMs, you would set --n-samples-per-event 100.")
parser.add_argument('--output', '-o', type=str, default="mass_lambdas.h5", help="Name of the h5 file to save with the samples.")
parser.add_argument('--scale-chirp', type=float, default=1.1, help="Factor to multiply by the standard deviation of the chirp mass Gaussian fit when drawing the mass proposal samples. Larger number increases support in the tails.")
parser.add_argument('--eos', type=str, default="GP", help="EoS model to use. Either 'UR' for the Universal Relation model, or 'GP' for the standard set of 10000 GP draws. Alternatively, an h5 file path can be provided and it is assumed that it follows the GP h5 file format.")
parser.add_argument('--eos-prior', type=str, default=None, help="Prior file for the UR model parameters. If not provided, will use a default.")
parser.add_argument('--n-eos', type=str, default="all", help="Number of EoSes to draw.")

args=parser.parse_args()
gmm_files = args.gw_gmm_files
output_file = args.output.split(".h5")[0]+".h5"
chirp_scale = args.scale_chirp
eos_type = args.eos
n_eos = args.n_eos

samples_per_event = args.n_samples_per_event
powerlaw = util_funcs.powerlaw

gw_gmm_files = np.loadtxt(gmm_files, dtype=str)
results = []
for file in gw_gmm_files:
    res = pickle.load(open(file, "rb")).result
    results.append(res)
    
medians = xp.array([np.median(res.posterior['chirp_mass_source']) for res in results])
stdevs = xp.array([np.std(res.posterior['chirp_mass_source']) for res in results])
medians = medians[:,None]
stdevs = stdevs[:,None]

stdevs *= chirp_scale


def get_GP_eoses_and_tovs_by_key(gp_file, n_samples):
    eoses = dict()
    mtovs = dict()
    with h5py.File(gp_file) as hh:
        if n_eos == "all":
            eos_keys = list(hh['ns'].keys())
        else:
            n_samples = int(n_samples)
            eos_keys = np.random.choice(list(hh['ns'].keys()), n_samples, replace=False) 
        for eos in tqdm(eos_keys):
            eoses[eos] = hh['ns'][eos][()]
            mtovs[eos] = max(hh['ns'][eos]['M'][()][np.gradient(eoses[eos]['M'], eoses[eos]['rhoc']) > 0])
    return eoses, mtovs

def get_GP_eoses_and_tovs_by_index(gp_file, n_samples):
    eoses = dict()
    mtovs = dict()
    with h5py.File(gp_file) as hh:
        if n_eos == "all":
            eos_indx = list(range(len(hh['ns'])))
        else:
            n_samples = int(n_samples)
            eos_indx = np.random.choice(len(hh['ns']), n_samples, replace=False) 
        for ii in tqdm(eos_indx):
            eos_key = f"eos_{ii}"
            eoses[eos_key] = hh['ns'][ii][()]
            mtovs[eos_key] = max(hh['ns'][ii]['M'][()][np.gradient(hh['ns'][ii]['M'][()], hh['ns'][ii]['rhoc'][()]) > 0])
    return eoses, mtovs

def get_UR_eoses_and_tovs(prior, n_samples):
    eoses=dict()
    mtovs=dict()
    for ii in range(n_samples):
        eos = f"eos_{ii}"
        sample = URprior.sample()
        eoses[eos] = [float(sample['m']), float(sample['b'])]
        mtovs[eos] = MTOV(eoses[eos]) 
    return eoses, mtovs

if eos_type.lower() == "UR":
    if not args.eos_prior and args.eos == "UR":
        print(u"No UR EoS Prior provided. Defaulting to:\n"
        "prior['m'] = bilby.core.prior.Uniform(name='m', minimum = -2.5, maximum=-0.5)\n"
        "prior['b'] = bilby.core.prior.Uniform(name='b', minimum = 2, maximum= 4)")
        URprior = bilby.core.prior.PriorDict()
        URprior['m'] = bilby.core.prior.Uniform(name='m', minimum = -2.5, maximum=-0.5)
        URprior['b'] = bilby.core.prior.Uniform(name='b', minimum = 2, maximum= 4)
    else:
        URprior = bilby.core.prior.PriorDict(args.eos_prior)
    eoses, mtovs = get_UR_eoses_and_tovs(URprior, n_eos)
else:
    if eos_type.lower() == "gp-astro":
        eosfile = '/home/isaac.legred/lwp/Examples/LCEHL_EOS_posterior_samples_PSR.h5'
    else:
        eosfile = eos_type
        
    try:
        eoses, mtovs = get_GP_eoses_and_tovs_by_key(eosfile, n_eos)
    except: 
        eoses, mtovs = get_GP_eoses_and_tovs_by_index(eosfile, n_eos)


def sample_population(N, params, medians=medians, stdevs=stdevs, N_results=len(results) ):
    samples=dict()
    chirp_samples = util_funcs.icdf_normal(xp.random.uniform(size=N//N_results), medians, stdevs).flatten()
    q_samples = util_funcs.icdf_powerlaw(xp.random.uniform(size=len(chirp_samples)), alpha = 1, minimum = 0.125, maximum=1)
    m1s, m2s = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(chirp_mass = chirp_samples, mass_ratio = q_samples)
    
    jacobian = chirp_samples / m1s**2 #d(M_c,q)/d(m1,m2)
    
    samples['mass_1_source'] = m1s
    samples['mass_2_source'] = m2s
    p_draw = util_funcs.pdf_normal(chirp_samples, medians, stdevs, 1/len(results)) * util_funcs.powerlaw(q_samples, alpha = 1, low = 0.125, high=1)
    samples['p_draw'] = p_draw
    samples['jacobian_d(M_c,q)/d(m1,m2)'] =jacobian  
    return samples


mass_samples = sample_population(samples_per_event * len(results), None)

masslambdas = dict()
masslambdas['masses'] = np.stack((mass_samples['mass_1_source'], mass_samples['mass_2_source']), axis=1)
masslambdas['masses'] = [(mm[0], mm[1]) for mm in masslambdas['masses']]
masslambdas['p_draw'] = mass_samples['p_draw'].tolist()
masslambdas['jacobian_d(M_c,q)/d(m1,m2)'] = mass_samples['jacobian_d(M_c,q)/d(m1,m2)'].tolist()

for eos in tqdm(eoses):
    m1macros = choose_macro_per_m(np.array([mass[0] for mass in masslambdas['masses']]), eoses[eos],
                                  {"Lambda": lambda x: np.zeros_like(x)})
    m2macros=choose_macro_per_m(np.array([mass[1] for mass in masslambdas['masses']]), eoses[eos],
                                  {"Lambda": lambda x: np.zeros_like(x)})
    lambdas = np.stack((m1macros['Lambda'], m2macros['Lambda']), axis=1)
    lambdas = [(ll[0], ll[1]) for ll in lambdas ]
    masslambdas[eos] = lambdas

with h5py.File(output_file, 'w') as ff:
    for key in masslambdas.keys():
        dset = ff.create_dataset(key, data=masslambdas[key])
        if 'eos' in key:
            dset.attrs['mtov'] = mtovs[key]
