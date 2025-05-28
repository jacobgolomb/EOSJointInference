import dill
import numpy as np
import bilby
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
import numpy as np
import sys
from sampling_utils import UtilFuncs
from radio_likelihoods import EMPulsarLikelihood
import models

parser = argparse.ArgumentParser()
parser.add_argument('--radio-data', '--data-path', type=str, help="path to h5 file with radio pulsar data")
parser.add_argument("--n-samples", "-n", type=int, default=5000, help="number of fiducial samples to draw from the population")
parser.add_argument('--samples-from-population', '-sp', type=str, help="Path to h5 file with the fiducial samples from the population (i.e. masses and lambdas)")
parser.add_argument('--proposal-result', '-p', type=str, help="Path to result file for the proposal hyperparameter")

parser.add_argument("--output-file", "-o", type=str, help="file to save", default="radio_pulsar_samples_with_weights.pkl")
parser.add_argument("--backend", "-b", type=str, default="jax", help="Backend to use. Either numpy or jax")

args = parser.parse_args()
output_file = args.output_file.split(".pkl")[0]+".pkl"
backend = args.backend
if backend.lower() == 'jax':
    import jax
    use_jax=True
    xp = jax.numpy
    util_funcs = UtilFuncs('jax')
    jax.device = jax.devices("gpu")
    jax.config.update("jax_enable_x64", True)
    models.set_backend("jax")

elif backend.lower() == 'numpy':
    xp = np
    import jax
    jax.device = jax.devices("cpu")[0]

    use_jax=False
    util_funcs = UtilFuncs('numpy')
    models.set_backend("numpy")
    
class RadioLikelihood(bilby.Likelihood):
    def __init__(self,radio_data_path, fiducial_population_params, backend='jax', n_samples=5000):

        self.radio_likelihood = EMPulsarLikelihood(path=radio_data_path, backend=backend)
        self.n_samples = n_samples
        self.parameters = dict( mmin=None, mpop=None, mu1=None, mu2=None, sigma1=None, sigma2=None, frac1=None, m=None, b=None)
        self.fiducial_parameters = fiducial_population_params
        self.draw_proposal_pulsars()
        self.xp = xp
        super().__init__(parameters = self.parameters)
        
    def draw_proposal_pulsars(self):
        self.proposal_pulsars =  models.draw_fiducial_pulsar_samples(seed=1, N=self.n_samples, **self.fiducial_parameters)
        self.proposal_radio_likelihoods = xp.asarray(self.radio_likelihood.likelihood_per_event_per_sample(self.proposal_pulsars['mass']))
    

likelihood = RadioLikelihood(radio_data_path=args.radio_data, n_samples=args.n_samples, 
                             backend=backend,fiducial_population_params={'mmin': 1, 'mmax': 3.2, 'mu': 1.35, 'sigma': 0.05, 'gaussian_frac': 0.4})
try:
    proposal_result = bilby.result.read_in_result(args.proposal_result)
    mass_params = proposal_result.posterior
except:
    try:
        proposal_result = pd.read_hdf(args.proposal_result)
    except:
        try:
            res = json.load(open(args.proposal_result, 'r'))
            proposal_result = pd.DataFrame(res['posterior']['content'])
        except:
            proposal_result = pd.read_csv(args.proposal_result)
        mass_params = proposal_result
powerlaw = util_funcs.powerlaw

n_pop_hyper_samples = len(mass_params)
mass_params = {key: xp.array(mass_params[key]).reshape((-1,1)) for key in mass_params if key not in ['m', 'b']}

def calc_log_like(mass_pop_weights, log_likelihood):
    per_event = []
    for ll_event in log_likelihood:
        like_term = ll_event + xp.log(mass_pop_weights)[:,None,:]
        per_event.append(xp.mean(xp.exp(like_term), axis=-1))
    return xp.sum(xp.log(xp.asarray(per_event)), axis=0) 

population_samples = h5py.File(args.samples_from_population)
mass_samples = xp.asarray(population_samples["mass"][()])
tovs = np.array([population_samples[eos].attrs['mtov'] for eos in population_samples if 'eos' in eos])

samples_with_weights = {}

pop_weights = models.get_pulsar_mass_weights(likelihood.proposal_pulsars['mass'], likelihood.proposal_pulsars['p_draw'], 
                            mass_params['mmin'], mass_params['mpop'], mass_params['mu1'], mass_params['mu2'], mass_params['sigma1'], mass_params['sigma2'], mass_params['frac1'])

selection = 1
samples_with_weights["mass_samples"] = {key: np.array(mass_params[key]).reshape((-1,1)) for key in mass_params}

eos_keys = [key for key in population_samples.keys() if 'eos_' in key]
loglike_array = np.asarray(calc_log_like(pop_weights, xp.log(likelihood.proposal_radio_likelihoods))) #This is independent of the EoS
for eos in tqdm(eos_keys):
    samples_with_weights[eos] = loglike_array

with open(output_file, "wb") as ww:
    pickle.dump(samples_with_weights, ww)