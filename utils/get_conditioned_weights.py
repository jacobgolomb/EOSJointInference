import pickle
import argparse
import numpy as np
import h5py
from scipy.stats import gaussian_kde

parser=argparse.ArgumentParser()
parser.add_argument("--weighted-samples", "-s", help="File with the samples with weights")
parser.add_argument("--samples-from-population", "-p", help="File with the samples from population. This is just needed because it is where mtovs are stored.")
parser.add_argument('--output-file', '-o', default="conditioned_weights.pkl", help="Where to save files")
parser.add_argument('--kde', action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

with open(args.weighted_samples, "rb") as rr:
    samples=pickle.load(rr)
    
population_samples = h5py.File(args.samples_from_population)
exclude = ["log_likelihood", "log_prior", "m", "b", "mmin", "Unnamed: 0"]
hyper_parameter_keys = [key for key in samples['mass_samples'] if key not in exclude]
print(hyper_parameter_keys)

if args.kde is True:
    proposal_array = np.array([samples['mass_samples'][key][()].squeeze() for key in hyper_parameter_keys])
    proposal_kde = gaussian_kde(proposal_array, bw_method=0.1)
    log_proposal_vals = np.log(proposal_kde(proposal_array))
    #kde_samples = proposal_kde.resample(10000)
    #kde_samples = {key: kde_samples[i, :] for i, key in enumerate(hyper_parameter_keys)}
    print("using KDE for proposals")
else: 
    log_proposal_vals=np.array(samples['mass_samples']['proposal_log_prob']).squeeze()
    print("using stored values for proposal")

loglike = np.array([samples[eos] for eos in samples if 'eos' in eos]).squeeze()
logweights = loglike - log_proposal_vals
tovs = np.array([population_samples[eos].attrs['mtov'] for eos in samples if 'eos' in eos])

mpops = samples['mass_samples']['mpop'].T * np.ones_like(logweights)
tovs = tovs[:,None] * np.ones_like(logweights)

good_condition = (mpops < tovs)

weights = dict()
weights['log_likelihoods'] = loglike + np.log(good_condition)
#weights['per_mass_population'] = np.average(np.exp(logweights_condition), axis=0)
#weights['per_mass_population_norm'] = np.sum(good_condition, axis=1)

#weights['per_mass_population'] = np.nan_to_num(np.sum(np.exp(logweights_condition), axis=0)/np.sum(good_condition, axis=0))
weights['log_proposal'] = log_proposal_vals
#weights['per_eos'] = np.average(np.exp(logweights_condition), axis=1)
#weights['per_eos_norm'] = np.sum(good_condition, axis=1)
#weights['per_eos'] =  np.nan_to_num(np.sum(np.exp(logweights_condition), axis=1)/weights['per_eos_norm'])
weights['eos_list'] = [eos for eos in samples if 'eos_' in eos]
weights['tov_list'] = np.array([population_samples[eos].attrs['mtov'] for eos in samples if 'eos' in eos])
weights['log_likelihoods_sorted'] = weights['log_likelihoods'][np.argsort(weights['eos_list']), :]
weights['mass_population_list'] = samples['mass_samples']

with open(args.output_file, 'wb') as ww:
    pickle.dump(weights,ww)